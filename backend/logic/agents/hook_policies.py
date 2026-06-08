"""Concrete hook implementations for the autopilot agent system.

Each hook handles one cross-cutting concern that previously lived as inline
code inside BaseAgent.run_llm_loop or AimlScientist._run_loop_iterations.

Register them all at once via default_hook_manager(), which is called from
AiAutopilot.__init__() in autopilot.py.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Generator

from .hooks import (
    Decision,
    Hook,
    HookContext,
    HookEvent,
    HookManager,
    HookOutcome,
)

if TYPE_CHECKING:
    from .base import AutopilotStep

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# TokenAccountingHook
# Replaces the duplicated response.usage blocks in both LLM loops.
# ──────────────────────────────────────────────────────────────────────────────


class TokenAccountingHook(Hook):
    """Track prompt/completion tokens and update ctx.agent_token_usage.

    Also writes 'last_prompt_tokens' into the usage dict so BaseAgent._step()
    can tag every AutopilotStep with the current context size.
    """

    name = "token_accounting"
    priority = 10
    events = frozenset({HookEvent.AFTER_LLM})

    def handle(self, hc: HookContext) -> Generator["AutopilotStep", None, HookOutcome]:
        yield from ()
        response = hc.response
        if response is None or not getattr(response, "usage", None):
            return HookOutcome.cont()

        u = hc.ctx.agent_token_usage.setdefault(
            hc.agent_name,
            {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "last_prompt_tokens": 0},
        )
        u["prompt_tokens"] += response.usage.prompt_tokens
        u["completion_tokens"] += response.usage.completion_tokens
        u["calls"] += 1
        u["last_prompt_tokens"] = response.usage.prompt_tokens

        log.debug(
            "token_accounting | agent=%s prompt=%d completion=%d total_calls=%d",
            hc.agent_name,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            u["calls"],
        )
        return HookOutcome.cont()


# ──────────────────────────────────────────────────────────────────────────────
# StopHook
# Replaces the ctx.should_stop checks at the top of each loop iteration.
# ──────────────────────────────────────────────────────────────────────────────


class StopHook(Hook):
    """Abort the LLM loop when a stop has been requested.

    Fires on BEFORE_LLM so the API call is never made after the user clicks
    Stop, instead of waiting for a potentially long blocking call to return.
    """

    name = "stop"
    priority = 1   # run first so abort beats everything
    events = frozenset({HookEvent.BEFORE_LLM})

    def handle(self, hc: HookContext) -> Generator["AutopilotStep", None, HookOutcome]:
        yield from ()
        if hc.ctx.should_stop:
            log.info(
                "StopHook | ABORT | agent=%s iteration=%s (stop requested)",
                hc.agent_name, hc.iteration,
            )
            return HookOutcome.abort(reason="stop requested")
        return HookOutcome.cont()


# ──────────────────────────────────────────────────────────────────────────────
# LoggingHook
# Consolidates key info/debug logs that were scattered across both loops.
# ──────────────────────────────────────────────────────────────────────────────


class LoggingHook(Hook):
    """Emit structured log entries for every major lifecycle event."""

    name = "logging"
    priority = 20
    events = frozenset({
        HookEvent.BEFORE_LLM,
        HookEvent.AFTER_LLM,
        HookEvent.BEFORE_TOOL,
        HookEvent.AFTER_TOOL,
        HookEvent.TOOL_ERROR,
        HookEvent.BEFORE_DELEGATE,
        HookEvent.AFTER_DELEGATE,
        HookEvent.RUN_START,
        HookEvent.RUN_END,
    })

    def handle(self, hc: HookContext) -> Generator["AutopilotStep", None, HookOutcome]:
        yield from ()
        ev = hc.event

        if ev is HookEvent.BEFORE_LLM:
            log.debug(
                "LLM call | agent=%s iteration=%s messages=%s",
                hc.agent_name, hc.iteration,
                len(hc.messages) if hc.messages else 0,
            )

        elif ev is HookEvent.AFTER_LLM:
            resp = hc.response
            choice = resp.choices[0] if resp and resp.choices else None
            log.debug(
                "LLM response | agent=%s finish_reason=%s has_tool_calls=%s",
                hc.agent_name,
                choice.finish_reason if choice else "?",
                bool(choice and choice.message.tool_calls) if choice else False,
            )

        elif ev is HookEvent.BEFORE_TOOL:
            raw = json.dumps(hc.args or {})
            log.info(
                "tool_call | agent=%s name=%s args=%s",
                hc.agent_name, hc.tool_name, raw[:600],
            )

        elif ev is HookEvent.AFTER_TOOL:
            content = hc.result
            if content is None:
                log.warning(
                    "tool_result | agent=%s name=%s returned None content",
                    hc.agent_name, hc.tool_name,
                )
            elif isinstance(content, list):
                log.debug(
                    "tool_result | agent=%s name=%s content=vision+text parts=%d",
                    hc.agent_name, hc.tool_name, len(content),
                )
            else:
                log.debug(
                    "tool_result | agent=%s name=%s content_len=%d",
                    hc.agent_name, hc.tool_name, len(content),
                )

        elif ev is HookEvent.TOOL_ERROR:
            log.error(
                "tool_error | agent=%s name=%s error=%s",
                hc.agent_name, hc.tool_name, hc.error,
            )

        elif ev is HookEvent.BEFORE_DELEGATE:
            log.info("Scientist delegating → %s", hc.label)

        elif ev is HookEvent.AFTER_DELEGATE:
            keys = list(hc.summary.keys()) if isinstance(hc.summary, dict) else type(hc.summary)
            log.info(
                "Delegate returned | label=%s summary_keys=%s", hc.label, keys
            )

        elif ev is HookEvent.RUN_START:
            log.info("Agent run start | agent=%s", hc.agent_name)

        elif ev is HookEvent.RUN_END:
            log.info("Agent run end | agent=%s", hc.agent_name)

        return HookOutcome.cont()


# ──────────────────────────────────────────────────────────────────────────────
# SteeringHook
# Replaces _steer_check / _delegate_with_steering in scientist.py.
# Fires on AFTER_DELEGATE; calls a lightweight eval LLM to decide whether to
# re-run the sub-agent.  Returns RETRY(new_instructions) or CONTINUE.
# ──────────────────────────────────────────────────────────────────────────────


class SteeringHook(Hook):
    """Review a sub-agent summary and optionally re-task it.

    This hook replicates the logic that was previously in AimlScientist._steer_check.
    It fires after every delegation and returns RETRY if the Scientist's eval LLM
    is not satisfied with the output.
    """

    name = "steering"
    priority = 50
    events = frozenset({HookEvent.AFTER_DELEGATE})

    def handle(self, hc: HookContext) -> Generator["AutopilotStep", None, HookOutcome]:
        from .base import AutopilotStep  # local import avoids circular refs at module level

        ctx = hc.ctx
        client = ctx.client
        deployment = ctx.deployment

        if client is None or not deployment:
            log.warning("SteeringHook | no client/deployment on ctx — skipping steering")
            yield from ()
            return HookOutcome.cont()

        instructions = hc.instructions or ""
        summary = hc.summary or {}
        label = hc.label or "Agent"

        eval_messages = [
            {
                "role": "system",
                "content": (
                    "You are the AIML Scientist reviewing a sub-agent's completed work. "
                    "Respond ONLY in JSON: "
                    "{\"satisfied\": bool, \"reason\": str, \"new_instructions\": string_or_null}. "
                    "Be satisfied unless there is a clear, specific, actionable problem "
                    "with the output that warrants re-running the agent."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Sub-agent: {label}\n"
                    f"Instructions given:\n{instructions}\n\n"
                    f"Agent summary (returned):\n{json.dumps(summary, indent=2)}\n\n"
                    f"Notebook context (cumulative findings):\n{ctx.notebook_text()}\n\n"
                    "Is this result sufficient to proceed? "
                    "If not, provide specific new_instructions."
                ),
            },
        ]
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=eval_messages,
                response_format={"type": "json_object"},
                max_completion_tokens=400,
            )
            # Attribute the eval call's tokens to the scientist.
            if response.usage:
                u = ctx.agent_token_usage.setdefault(
                    "scientist",
                    {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "last_prompt_tokens": 0},
                )
                u["prompt_tokens"] += response.usage.prompt_tokens
                u["completion_tokens"] += response.usage.completion_tokens
                u["calls"] += 1
                u["last_prompt_tokens"] = response.usage.prompt_tokens

            decision = json.loads(response.choices[0].message.content or "{}")
        except Exception as exc:
            log.warning("SteeringHook | eval LLM failed: %s — assuming satisfied", exc)
            yield from ()
            return HookOutcome.cont()

        if decision.get("satisfied", True):
            log.info("SteeringHook | agent=%s satisfied=True", label)
            yield from ()
            return HookOutcome.cont()

        reason = decision.get("reason", "")
        new_instructions = decision.get("new_instructions") or instructions
        log.info(
            "SteeringHook | agent=%s satisfied=False reason=%r",
            label, reason[:120],
        )

        # Yield a thought step visible in the UI (replicates prior behaviour).
        from .base import AgentContext  # already imported, but keep it safe
        thought = AutopilotStep(
            index=ctx.next_step_index(),
            kind="thought",
            title=f"[Scientist] Re-tasking {label}",
            detail=f"**Not satisfied:** {reason}\n\n**Updated instructions:** {new_instructions}",
            agent="scientist",
            phase=ctx.current_phase,
        )
        yield thought
        return HookOutcome.retry(instructions=new_instructions, reason=reason)


# ──────────────────────────────────────────────────────────────────────────────
# ModelTesterGateHook
# Replaces the hardcoded "run Model Tester before Review" block in scientist.py.
# Fires on BEFORE_DELEGATE; when label=="Review", auto-runs ModelTesterAgent on
# any untested runs before letting the delegation proceed.
# ──────────────────────────────────────────────────────────────────────────────


class ModelTesterGateHook(Hook):
    """Ensure Model Tester has run on all training runs before Review.

    Replicates the logic that was hardcoded in AimlScientist._dispatch for the
    delegate_to_review branch.  Any untested runs (those with test_data_path but
    not in ctx.tested_run_ids) are evaluated here before Review is delegated.
    """

    name = "model_tester_gate"
    priority = 30   # run before SteeringHook (priority 50)
    events = frozenset({HookEvent.BEFORE_DELEGATE})

    def handle(self, hc: HookContext) -> Generator["AutopilotStep", None, HookOutcome]:
        if hc.label != "Review":
            yield from ()
            return HookOutcome.cont()

        ctx = hc.ctx
        client = ctx.client
        deployment = ctx.deployment

        if client is None or not deployment:
            log.warning("ModelTesterGateHook | no client/deployment — skipping gate")
            yield from ()
            return HookOutcome.cont()

        untested = [
            r for r in ctx.training_runs
            if r.get("test_data_path") and r.get("run_id") not in ctx.tested_run_ids
        ]

        if not untested:
            yield from ()
            return HookOutcome.cont()

        log.info(
            "ModelTesterGateHook | auto-running Model Tester before Review | untested=%s",
            [r["run_id"] for r in untested],
        )

        # Yield an auto-routing thought step so the UI still shows this action.
        auto_step = _make_step(
            ctx,
            kind="thought",
            title="AIML Scientist — Auto-routing",
            detail=f"Running Model Tester on {len(untested)} untested run(s) before Review.",
            agent="scientist",
        )
        yield auto_step

        # Lazy import to avoid circular dep at module level.
        from .model_tester import ModelTesterAgent

        tester = ModelTesterAgent(client, deployment, ctx)
        tester_summary = yield from tester.run(
            "Evaluate all trained models on the held-out test set."
        )
        for r in untested:
            ctx.tested_run_ids.add(r["run_id"])

        log.info(
            "ModelTesterGateHook | tester finished | runs_evaluated=%s",
            tester_summary.get("runs_evaluated", 0) if isinstance(tester_summary, dict) else "?",
        )

        return HookOutcome.cont()


# ──────────────────────────────────────────────────────────────────────────────
# GuardrailHook  (NEW capability — no equivalent existed before)
# Fires on BEFORE_TOOL; can modify or skip dangerous/invalid tool calls.
# ──────────────────────────────────────────────────────────────────────────────


class GuardrailHook(Hook):
    """Validate or veto tool calls before they are dispatched.

    This hook is intentionally minimal out of the box — it ships with a small set
    of example rules.  Extend check_args() to add domain-specific constraints.

    Returns:
        CONTINUE  — call is safe, proceed normally.
        MODIFY    — call is safe but args were cleaned/normalised.
        SKIP      — call is disallowed; canned error result injected.
    """

    name = "guardrail"
    priority = 5   # run very early, before logging even
    events = frozenset({HookEvent.BEFORE_TOOL})

    # Tool names that are never allowed (example: destructive ops).
    BLOCKED_TOOLS: frozenset[str] = frozenset()

    # Maximum number of columns that feature-engineering ops may reference at once.
    MAX_COLUMNS_PER_OP: int = 50

    def handle(self, hc: HookContext) -> Generator["AutopilotStep", None, HookOutcome]:
        yield from ()
        return self.check_args(hc.tool_name or "", hc.args or {})

    def check_args(self, tool_name: str, args: dict) -> HookOutcome:
        """Override this method to add custom validation rules."""

        if tool_name in self.BLOCKED_TOOLS:
            log.warning("GuardrailHook | BLOCKED tool=%s", tool_name)
            return HookOutcome.skip(
                result=json.dumps({"error": f"Tool '{tool_name}' is not permitted."}),
                reason=f"blocked tool: {tool_name}",
            )

        # Guard against accidentally passing huge column lists.
        columns = args.get("columns") or args.get("params", {}).get("columns", [])
        if isinstance(columns, list) and len(columns) > self.MAX_COLUMNS_PER_OP:
            log.warning(
                "GuardrailHook | truncating oversized columns list | tool=%s len=%d",
                tool_name, len(columns),
            )
            trimmed = dict(args)
            if "columns" in trimmed:
                trimmed["columns"] = columns[: self.MAX_COLUMNS_PER_OP]
            elif "params" in trimmed and "columns" in trimmed["params"]:
                trimmed["params"] = dict(trimmed["params"])
                trimmed["params"]["columns"] = columns[: self.MAX_COLUMNS_PER_OP]
            return HookOutcome.modify(
                args=trimmed,
                reason=f"columns list truncated to {self.MAX_COLUMNS_PER_OP}",
            )

        return HookOutcome.cont()


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────


def default_hook_manager() -> HookManager:
    """Build and return the default HookManager with all standard policies.

    Called once by AiAutopilot.__init__() and stored on AgentContext.hooks.
    """
    manager = HookManager()
    manager.register(StopHook())
    manager.register(GuardrailHook())
    manager.register(TokenAccountingHook())
    manager.register(LoggingHook())
    manager.register(ModelTesterGateHook())
    manager.register(SteeringHook())
    return manager


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_step(
    ctx: Any,
    kind: str,
    title: str,
    detail: str = "",
    agent: str = "scientist",
) -> "AutopilotStep":
    """Construct an AutopilotStep without an agent instance reference."""
    from .base import AutopilotStep

    last_tokens = ctx.agent_token_usage.get(agent, {}).get("last_prompt_tokens")
    data = {"context_tokens": last_tokens} if last_tokens is not None else None
    return AutopilotStep(
        index=ctx.next_step_index(),
        kind=kind,
        title=title,
        detail=detail,
        data=data,
        agent=agent,
        phase=ctx.current_phase,
    )
