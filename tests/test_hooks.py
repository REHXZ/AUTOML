"""Tests for the hook lifecycle framework and policies.

Uses a minimal in-memory fake OpenAI client (no network calls) so the tests
run offline and deterministically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.logic.agents.base import AgentContext, AutopilotStep, BaseAgent
from backend.logic.agents.hooks import (
    Decision,
    Hook,
    HookContext,
    HookEvent,
    HookManager,
    HookOutcome,
)
from backend.logic.agents.hook_policies import (
    GuardrailHook,
    ModelTesterGateHook,
    SteeringHook,
    StopHook,
    TokenAccountingHook,
    default_hook_manager,
)
from backend.services.project_store import ProjectStore


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ctx(**kwargs) -> AgentContext:
    """Create a minimal AgentContext suitable for hook tests."""
    store = MagicMock(spec=ProjectStore)
    ctx = AgentContext(project_id="proj-1", store=store, **kwargs)
    return ctx


def _drain(gen):
    """Exhaust a generator and return (list_of_yielded, return_value)."""
    items = []
    try:
        while True:
            items.append(next(gen))
    except StopIteration as exc:
        return items, exc.value


def _fire(manager: HookManager, hc: HookContext):
    """Fire a hook event and return (steps_yielded, outcome)."""
    gen = manager.fire(hc)
    return _drain(gen)


# ──────────────────────────────────────────────────────────────────────────────
# HookOutcome merge
# ──────────────────────────────────────────────────────────────────────────────


class TestHookOutcomeMerge:
    def test_abort_beats_everything(self):
        a = HookOutcome.abort("stop")
        assert a.merge(HookOutcome.cont()).decision is Decision.ABORT
        assert HookOutcome.cont().merge(a).decision is Decision.ABORT
        assert HookOutcome.retry().merge(a).decision is Decision.ABORT

    def test_retry_beats_skip_modify_continue(self):
        r = HookOutcome.retry()
        assert HookOutcome.cont().merge(r).decision is Decision.RETRY
        assert HookOutcome.modify().merge(r).decision is Decision.RETRY
        assert HookOutcome.skip().merge(r).decision is Decision.RETRY

    def test_skip_beats_modify_continue(self):
        s = HookOutcome.skip(result="cached")
        assert HookOutcome.cont().merge(s).decision is Decision.SKIP
        assert HookOutcome.modify().merge(s).decision is Decision.SKIP

    def test_modify_beats_continue(self):
        m = HookOutcome.modify(args={"a": 1})
        assert HookOutcome.cont().merge(m).decision is Decision.MODIFY

    def test_two_modifies_merge_fields(self):
        m1 = HookOutcome.modify(args={"a": 1})
        m2 = HookOutcome.modify(instructions="new instr")
        merged = m1.merge(m2)
        assert merged.decision is Decision.MODIFY
        assert merged.args == {"a": 1}
        assert merged.instructions == "new instr"

    def test_continue_is_identity(self):
        c = HookOutcome.cont()
        assert c.merge(HookOutcome.cont()).decision is Decision.CONTINUE


# ──────────────────────────────────────────────────────────────────────────────
# HookManager — registration and firing
# ──────────────────────────────────────────────────────────────────────────────


class TestHookManager:
    def test_empty_manager_returns_continue(self):
        manager = HookManager()
        ctx = _ctx()
        hc = HookContext(HookEvent.BEFORE_TOOL, "test_agent", ctx)
        steps, outcome = _fire(manager, hc)
        assert steps == []
        assert outcome.decision is Decision.CONTINUE

    def test_hook_not_matching_event_is_skipped(self):
        fired = []

        class WrongEventHook(Hook):
            name = "wrong"
            events = frozenset({HookEvent.AFTER_LLM})

            def handle(self, hc):
                fired.append(True)
                yield from ()
                return HookOutcome.cont()

        manager = HookManager()
        manager.register(WrongEventHook())
        ctx = _ctx()
        hc = HookContext(HookEvent.BEFORE_TOOL, "agent", ctx)
        _fire(manager, hc)
        assert fired == []

    def test_hooks_run_in_priority_order(self):
        order = []

        class H(Hook):
            def __init__(self, p, label):
                self.priority = p
                self.name = label
                self.events = frozenset({HookEvent.BEFORE_TOOL})

            def handle(self, hc):
                order.append(self.name)
                yield from ()
                return HookOutcome.cont()

        manager = HookManager()
        manager.register(H(30, "c"))
        manager.register(H(10, "a"))
        manager.register(H(20, "b"))

        ctx = _ctx()
        hc = HookContext(HookEvent.BEFORE_TOOL, "agent", ctx)
        _fire(manager, hc)
        assert order == ["a", "b", "c"]

    def test_abort_stops_remaining_hooks(self):
        ran = []

        class AbortHook(Hook):
            name = "aborter"
            priority = 10
            events = frozenset({HookEvent.BEFORE_TOOL})

            def handle(self, hc):
                yield from ()
                return HookOutcome.abort("test abort")

        class AfterHook(Hook):
            name = "after"
            priority = 20
            events = frozenset({HookEvent.BEFORE_TOOL})

            def handle(self, hc):
                ran.append(True)
                yield from ()
                return HookOutcome.cont()

        manager = HookManager()
        manager.register(AbortHook())
        manager.register(AfterHook())

        ctx = _ctx()
        hc = HookContext(HookEvent.BEFORE_TOOL, "agent", ctx)
        _, outcome = _fire(manager, hc)
        assert outcome.decision is Decision.ABORT
        assert ran == []

    def test_hook_that_yields_steps_emits_them(self):
        class StepHook(Hook):
            name = "step_emitter"
            priority = 10
            events = frozenset({HookEvent.BEFORE_TOOL})

            def handle(self, hc):
                yield AutopilotStep(
                    index=1, kind="thought", title="from hook", agent="test", phase="modeling"
                )
                return HookOutcome.cont()

        manager = HookManager()
        manager.register(StepHook())
        ctx = _ctx()
        hc = HookContext(HookEvent.BEFORE_TOOL, "agent", ctx)
        steps, outcome = _fire(manager, hc)
        assert len(steps) == 1
        assert steps[0].title == "from hook"
        assert outcome.decision is Decision.CONTINUE

    def test_hook_exception_is_swallowed_and_continues(self):
        class BadHook(Hook):
            name = "bad"
            priority = 10
            events = frozenset({HookEvent.BEFORE_TOOL})

            def handle(self, hc):
                yield from ()
                raise ValueError("oops")

        class GoodHook(Hook):
            name = "good"
            priority = 20
            events = frozenset({HookEvent.BEFORE_TOOL})

            def handle(self, hc):
                yield from ()
                return HookOutcome.modify(args={"safe": True})

        manager = HookManager()
        manager.register(BadHook())
        manager.register(GoodHook())
        ctx = _ctx()
        hc = HookContext(HookEvent.BEFORE_TOOL, "agent", ctx)
        _, outcome = _fire(manager, hc)
        assert outcome.decision is Decision.MODIFY


# ──────────────────────────────────────────────────────────────────────────────
# TokenAccountingHook
# ──────────────────────────────────────────────────────────────────────────────


class TestTokenAccountingHook:
    def _fake_response(self, prompt_tokens=100, completion_tokens=50):
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        resp = MagicMock()
        resp.usage = usage
        return resp

    def test_accumulates_tokens(self):
        hook = TokenAccountingHook()
        ctx = _ctx()
        manager = HookManager()
        manager.register(hook)

        resp = self._fake_response(100, 50)
        hc = HookContext(HookEvent.AFTER_LLM, "eda", ctx, response=resp)
        _fire(manager, hc)

        u = ctx.agent_token_usage["eda"]
        assert u["prompt_tokens"] == 100
        assert u["completion_tokens"] == 50
        assert u["calls"] == 1
        assert u["last_prompt_tokens"] == 100

    def test_accumulates_across_calls(self):
        hook = TokenAccountingHook()
        ctx = _ctx()
        manager = HookManager()
        manager.register(hook)

        for _ in range(3):
            resp = self._fake_response(200, 80)
            hc = HookContext(HookEvent.AFTER_LLM, "modeling", ctx, response=resp)
            _fire(manager, hc)

        u = ctx.agent_token_usage["modeling"]
        assert u["prompt_tokens"] == 600
        assert u["completion_tokens"] == 240
        assert u["calls"] == 3

    def test_no_usage_is_noop(self):
        hook = TokenAccountingHook()
        ctx = _ctx()
        manager = HookManager()
        manager.register(hook)

        resp = MagicMock()
        resp.usage = None
        hc = HookContext(HookEvent.AFTER_LLM, "eda", ctx, response=resp)
        _fire(manager, hc)
        assert ctx.agent_token_usage == {}


# ──────────────────────────────────────────────────────────────────────────────
# StopHook
# ──────────────────────────────────────────────────────────────────────────────


class TestStopHook:
    def test_aborts_when_stop_set(self):
        ctx = _ctx()
        ctx.should_stop = True
        hook = StopHook()
        manager = HookManager()
        manager.register(hook)
        hc = HookContext(HookEvent.BEFORE_LLM, "scientist", ctx, iteration=5)
        _, outcome = _fire(manager, hc)
        assert outcome.decision is Decision.ABORT

    def test_continues_when_stop_not_set(self):
        ctx = _ctx()
        ctx.should_stop = False
        hook = StopHook()
        manager = HookManager()
        manager.register(hook)
        hc = HookContext(HookEvent.BEFORE_LLM, "scientist", ctx, iteration=0)
        _, outcome = _fire(manager, hc)
        assert outcome.decision is Decision.CONTINUE


# ──────────────────────────────────────────────────────────────────────────────
# GuardrailHook
# ──────────────────────────────────────────────────────────────────────────────


class TestGuardrailHook:
    def test_blocked_tool_returns_skip(self):
        hook = GuardrailHook()
        hook.BLOCKED_TOOLS = frozenset({"dangerous_delete"})
        ctx = _ctx()
        manager = HookManager()
        manager.register(hook)

        hc = HookContext(
            HookEvent.BEFORE_TOOL, "fe", ctx, tool_name="dangerous_delete", args={}
        )
        _, outcome = _fire(manager, hc)
        assert outcome.decision is Decision.SKIP
        assert "not permitted" in outcome.result

    def test_oversized_columns_list_is_trimmed(self):
        hook = GuardrailHook()
        hook.MAX_COLUMNS_PER_OP = 5
        ctx = _ctx()
        manager = HookManager()
        manager.register(hook)

        big_args = {"columns": list(range(20)), "operation": "standard_scale"}
        hc = HookContext(HookEvent.BEFORE_TOOL, "fe", ctx, tool_name="scale", args=big_args)
        _, outcome = _fire(manager, hc)
        assert outcome.decision is Decision.MODIFY
        assert len(outcome.args["columns"]) == 5

    def test_safe_tool_passes_through(self):
        hook = GuardrailHook()
        ctx = _ctx()
        manager = HookManager()
        manager.register(hook)

        hc = HookContext(
            HookEvent.BEFORE_TOOL, "fe", ctx, tool_name="standard_scale",
            args={"columns": ["a", "b"]}
        )
        _, outcome = _fire(manager, hc)
        assert outcome.decision is Decision.CONTINUE

    def test_nested_params_columns_also_trimmed(self):
        hook = GuardrailHook()
        hook.MAX_COLUMNS_PER_OP = 3
        ctx = _ctx()
        manager = HookManager()
        manager.register(hook)

        nested_args = {"params": {"columns": list(range(10))}}
        hc = HookContext(
            HookEvent.BEFORE_TOOL, "fe", ctx, tool_name="drop_cols", args=nested_args
        )
        _, outcome = _fire(manager, hc)
        assert outcome.decision is Decision.MODIFY
        assert len(outcome.args["params"]["columns"]) == 3


# ──────────────────────────────────────────────────────────────────────────────
# SteeringHook
# ──────────────────────────────────────────────────────────────────────────────


class _FakeSteeringClient:
    """Fake OpenAI client that returns a sequence of canned responses."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self._idx = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        resp_data = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        choice = MagicMock()
        choice.message.content = json.dumps(resp_data)
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = None
        return resp


class TestSteeringHook:
    def test_satisfied_returns_continue(self):
        client = _FakeSteeringClient([{"satisfied": True, "reason": "ok", "new_instructions": None}])
        ctx = _ctx()
        ctx.client = client
        ctx.deployment = "gpt-4"

        hook = SteeringHook()
        manager = HookManager()
        manager.register(hook)

        hc = HookContext(
            HookEvent.AFTER_DELEGATE, "scientist", ctx,
            label="EDA", instructions="do eda", summary={"ok": True},
        )
        steps, outcome = _fire(manager, hc)
        assert outcome.decision is Decision.CONTINUE
        assert steps == []

    def test_unsatisfied_returns_retry_with_thought_step(self):
        client = _FakeSteeringClient([{
            "satisfied": False,
            "reason": "missing charts",
            "new_instructions": "create distribution charts",
        }])
        ctx = _ctx()
        ctx.client = client
        ctx.deployment = "gpt-4"

        hook = SteeringHook()
        manager = HookManager()
        manager.register(hook)

        hc = HookContext(
            HookEvent.AFTER_DELEGATE, "scientist", ctx,
            label="EDA", instructions="do eda", summary={},
        )
        steps, outcome = _fire(manager, hc)
        assert outcome.decision is Decision.RETRY
        assert outcome.instructions == "create distribution charts"
        assert len(steps) == 1
        assert "Re-tasking" in steps[0].title

    def test_no_client_is_noop(self):
        ctx = _ctx()  # client is None

        hook = SteeringHook()
        manager = HookManager()
        manager.register(hook)

        hc = HookContext(
            HookEvent.AFTER_DELEGATE, "scientist", ctx,
            label="EDA", instructions="do eda", summary={},
        )
        _, outcome = _fire(manager, hc)
        assert outcome.decision is Decision.CONTINUE

    def test_eval_llm_failure_is_noop(self):
        class BrokenClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        raise RuntimeError("network error")

        ctx = _ctx()
        ctx.client = BrokenClient()
        ctx.deployment = "gpt-4"

        hook = SteeringHook()
        manager = HookManager()
        manager.register(hook)

        hc = HookContext(
            HookEvent.AFTER_DELEGATE, "scientist", ctx,
            label="EDA", instructions="do eda", summary={},
        )
        _, outcome = _fire(manager, hc)
        assert outcome.decision is Decision.CONTINUE


# ──────────────────────────────────────────────────────────────────────────────
# default_hook_manager
# ──────────────────────────────────────────────────────────────────────────────


class TestDefaultHookManager:
    def test_all_policies_registered(self):
        manager = default_hook_manager()
        names = {h.name for h in manager.registered_hooks()}
        assert "token_accounting" in names
        assert "stop" in names
        assert "logging" in names
        assert "steering" in names
        assert "model_tester_gate" in names
        assert "guardrail" in names

    def test_stop_hook_has_lowest_priority(self):
        manager = default_hook_manager()
        hooks = manager.registered_hooks()
        stop = next(h for h in hooks if h.name == "stop")
        assert all(stop.priority <= h.priority for h in hooks)


# ──────────────────────────────────────────────────────────────────────────────
# Integration: BaseAgent.run_llm_loop with hooks
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeChoice:
    finish_reason: str
    message: Any


class _FakeLLMClient:
    """Returns one tool_call response then a stop response."""

    def __init__(self, tool_calls=None, final_content="Done"):
        self._calls = 0
        self._tool_calls = tool_calls or []
        self._final_content = final_content
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self._calls += 1
        if self._calls == 1 and self._tool_calls:
            msg = MagicMock()
            msg.content = None
            msg.tool_calls = self._tool_calls
            usage = MagicMock()
            usage.prompt_tokens = 50
            usage.completion_tokens = 20
            resp = MagicMock()
            resp.choices = [MagicMock(finish_reason="tool_calls", message=msg)]
            resp.usage = usage
            return resp
        msg = MagicMock()
        msg.content = self._final_content
        msg.tool_calls = None
        usage = MagicMock()
        usage.prompt_tokens = 30
        usage.completion_tokens = 10
        resp = MagicMock()
        resp.choices = [MagicMock(finish_reason="stop", message=msg)]
        resp.usage = usage
        return resp


class _SimpleAgent(BaseAgent):
    """Minimal concrete agent for integration tests."""

    name = "simple"
    display_name = "Simple"

    def __init__(self, client, deployment, context):
        super().__init__(client, deployment, context)
        self._dispatched: list[str] = []

    def _dispatch(self, name, args, tool_call_id):
        self._dispatched.append(name)
        return json.dumps({"done": True}), None, True  # terminate after first tool call

    def run(self, instructions=""):
        yield self._step("agent_start", "Simple dispatched", instructions)
        tools = [{"type": "function", "function": {"name": "dummy", "parameters": {}}}]
        yield from self.run_llm_loop(
            system_prompt="You are a test agent.",
            user_prompt=instructions,
            tools=tools,
            dispatch=self._dispatch,
        )
        yield self._step("agent_end", "Simple finished", "")
        return {"dispatched": self._dispatched}


class TestRunLlmLoopIntegration:
    def _make_tool_call(self, name="dummy", args="{}"):
        tc = MagicMock()
        tc.function.name = name
        tc.function.arguments = args
        tc.id = "tc-1"
        tc.model_dump.return_value = {"function": {"name": name, "arguments": args}}
        return tc

    def test_tool_dispatched_and_steps_yielded(self):
        ctx = _ctx()
        tc = self._make_tool_call("dummy")
        client = _FakeLLMClient(tool_calls=[tc])
        agent = _SimpleAgent(client, "gpt-4", ctx)

        steps, summary = _drain(agent.run("test"))
        kinds = [s.kind for s in steps]
        assert "agent_start" in kinds
        assert "tool_call" in kinds
        assert "agent_end" in kinds
        assert summary["dispatched"] == ["dummy"]

    def test_stop_hook_aborts_loop(self):
        ctx = _ctx()
        ctx.should_stop = True
        ctx.hooks = HookManager()
        ctx.hooks.register(StopHook())

        tc = self._make_tool_call("dummy")
        client = _FakeLLMClient(tool_calls=[tc])
        agent = _SimpleAgent(client, "gpt-4", ctx)

        steps, summary = _drain(agent.run("test"))
        # Loop aborted before making any tool calls.
        assert "tool_call" not in [s.kind for s in steps]

    def test_guardrail_skip_short_circuits_dispatch(self):
        ctx = _ctx()
        guard = GuardrailHook()
        guard.BLOCKED_TOOLS = frozenset({"dummy"})
        ctx.hooks = HookManager()
        ctx.hooks.register(guard)

        tc = self._make_tool_call("dummy")
        client = _FakeLLMClient(tool_calls=[tc])
        agent = _SimpleAgent(client, "gpt-4", ctx)

        steps, summary = _drain(agent.run("test"))
        # Tool was skipped — dispatch was never called.
        assert summary["dispatched"] == []

    def test_token_accounting_via_hook(self):
        ctx = _ctx()
        ctx.hooks = HookManager()
        ctx.hooks.register(TokenAccountingHook())

        tc = self._make_tool_call("dummy")
        client = _FakeLLMClient(tool_calls=[tc])
        agent = _SimpleAgent(client, "gpt-4", ctx)

        _drain(agent.run("test"))
        u = ctx.agent_token_usage.get("simple", {})
        assert u.get("calls", 0) >= 1
        assert u.get("prompt_tokens", 0) > 0
