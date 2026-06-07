"""Hook lifecycle framework for the autopilot agent system.

This module defines the core hook infrastructure — events, outcomes, the Hook base
class, and HookManager. It intentionally has NO imports from other agent modules to
avoid circular dependencies. Concrete hook implementations live in hook_policies.py.

Usage
-----
    # In autopilot.py — wire up a manager and register policies:
    from backend.logic.agents.hooks import HookManager
    from backend.logic.agents.hook_policies import default_hook_manager
    ctx.hooks = default_hook_manager()

    # Anywhere inside an agent loop — fire an event:
    outcome = yield from self._fire(HookContext(HookEvent.BEFORE_TOOL, ...))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from .base import AgentContext, AutopilotStep

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Events
# ──────────────────────────────────────────────────────────────────────────────


class HookEvent(str, Enum):
    """All lifecycle events the hook system can fire."""

    RUN_START = "run_start"
    """Fired once when an agent's .run() begins."""

    RUN_END = "run_end"
    """Fired once when an agent's .run() returns (including on error path)."""

    BEFORE_LLM = "before_llm"
    """Fired before every chat.completions.create() call. ABORT skips the call."""

    AFTER_LLM = "after_llm"
    """Fired after a successful chat.completions.create(). Carries raw response."""

    BEFORE_TOOL = "before_tool"
    """Fired before dispatch(name, args) is called.
    MODIFY can rewrite args. SKIP short-circuits dispatch entirely."""

    AFTER_TOOL = "after_tool"
    """Fired after dispatch() returns. MODIFY can rewrite tool_content."""

    TOOL_ERROR = "tool_error"
    """Fired when dispatch() raises. SKIP provides a canned tool_content recovery."""

    BEFORE_DELEGATE = "before_delegate"
    """Fired before the Scientist spawns a sub-agent.
    MODIFY can rewrite instructions. SKIP returns canned summary."""

    AFTER_DELEGATE = "after_delegate"
    """Fired after a sub-agent's .run() returns.
    RETRY re-runs the agent with (optionally new) instructions."""

    STEP_EMITTED = "step_emitted"
    """Fired for every AutopilotStep that passes through _tee. Observe-only."""

    ASK_USER = "ask_user"
    """Fired when the Scientist emits an 'ask' step to pause for user input."""


# ──────────────────────────────────────────────────────────────────────────────
# Outcome
# ──────────────────────────────────────────────────────────────────────────────


class Decision(str, Enum):
    """Flow-control decision a hook can return."""

    CONTINUE = "continue"
    """Do nothing special — proceed normally."""

    MODIFY = "modify"
    """Change something (args / instructions / result) but continue executing."""

    SKIP = "skip"
    """Don't execute the action; use the provided result instead."""

    RETRY = "retry"
    """Re-run the current action (delegation only), optionally with new instructions."""

    ABORT = "abort"
    """Terminate the enclosing loop immediately."""


# Precedence for merge(): higher index wins.
_DECISION_PRECEDENCE = {
    Decision.CONTINUE: 0,
    Decision.MODIFY: 1,
    Decision.SKIP: 2,
    Decision.RETRY: 3,
    Decision.ABORT: 4,
}


@dataclass
class HookOutcome:
    """What a hook returns to the agent loop to control flow."""

    decision: Decision = Decision.CONTINUE

    # For MODIFY: new args dict (tool) or new instructions string (delegation).
    args: dict[str, Any] | None = None
    instructions: str | None = None

    # For SKIP (tool) or RETRY (delegation): the canned result / new instructions.
    result: Any = None

    # Human-readable reason surfaced in logs / thought steps.
    reason: str = ""

    # ── factories ──

    @classmethod
    def cont(cls) -> "HookOutcome":
        return cls(Decision.CONTINUE)

    @classmethod
    def modify(
        cls,
        args: dict | None = None,
        instructions: str | None = None,
        reason: str = "",
    ) -> "HookOutcome":
        return cls(Decision.MODIFY, args=args, instructions=instructions, reason=reason)

    @classmethod
    def skip(cls, result: Any = None, reason: str = "") -> "HookOutcome":
        return cls(Decision.SKIP, result=result, reason=reason)

    @classmethod
    def retry(cls, instructions: str | None = None, reason: str = "") -> "HookOutcome":
        return cls(Decision.RETRY, instructions=instructions, reason=reason)

    @classmethod
    def abort(cls, reason: str = "") -> "HookOutcome":
        return cls(Decision.ABORT, reason=reason)

    def merge(self, other: "HookOutcome") -> "HookOutcome":
        """Return whichever outcome has higher precedence.

        Equal precedence: keep self (first-registered hook wins on ties).
        For MODIFY, we accumulate: if both are MODIFY and both carry separate
        fields, combine them so multiple MODIFY hooks can each patch a subset.
        """
        my_rank = _DECISION_PRECEDENCE[self.decision]
        other_rank = _DECISION_PRECEDENCE[other.decision]

        if other_rank > my_rank:
            return other

        # Both MODIFY — merge their fields (other's non-None fields override self's).
        if self.decision is Decision.MODIFY and other.decision is Decision.MODIFY:
            return HookOutcome(
                decision=Decision.MODIFY,
                args=other.args if other.args is not None else self.args,
                instructions=other.instructions if other.instructions is not None else self.instructions,
                result=other.result if other.result is not None else self.result,
                reason=other.reason or self.reason,
            )

        return self


# ──────────────────────────────────────────────────────────────────────────────
# Event payload
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class HookContext:
    """Payload passed to every hook.handle() call.

    Only the fields relevant to the specific event are populated; the rest are None.
    """

    event: HookEvent
    agent_name: str
    ctx: "AgentContext"

    # BEFORE_TOOL / AFTER_TOOL / TOOL_ERROR
    tool_name: str | None = None
    args: dict[str, Any] | None = None
    result: Any = None          # tool_content after dispatch, or error recovery

    # BEFORE_DELEGATE / AFTER_DELEGATE
    label: str | None = None    # e.g. "EDA", "Review"
    instructions: str | None = None
    summary: dict | None = None # sub-agent return value (AFTER_DELEGATE)

    # BEFORE_LLM / AFTER_LLM
    messages: list[dict] | None = None
    iteration: int | None = None
    response: Any = None        # raw OpenAI response object

    # STEP_EMITTED
    step: "AutopilotStep | None" = None

    # TOOL_ERROR
    error: Exception | None = None

    # Arbitrary extras for hooks that need additional data.
    extra: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Hook base
# ──────────────────────────────────────────────────────────────────────────────


class Hook:
    """Base class for all lifecycle hooks.

    Subclass this and override handle() to add behaviour.  handle() is a
    generator so hooks can yield AutopilotStep objects (e.g. a "thought" step)
    before returning their HookOutcome.

    Attributes
    ----------
    name:     Human-readable identifier (for logging).
    priority: Lower value = runs earlier. Default 100.
    events:   Set of HookEvent values this hook wants to receive.
              Empty set means the hook will never be called.
    """

    name: str = "hook"
    priority: int = 100
    events: frozenset[HookEvent] = frozenset()

    def handle(
        self, hc: HookContext
    ) -> "Generator[AutopilotStep, None, HookOutcome]":
        """Process an event.  Yield AutopilotStep objects as needed, then return
        a HookOutcome.  The default implementation is a no-op."""
        yield from ()
        return HookOutcome.cont()


# ──────────────────────────────────────────────────────────────────────────────
# Manager
# ──────────────────────────────────────────────────────────────────────────────


class HookManager:
    """Registry and dispatcher for Hook instances.

    Register hooks with .register(); fire them with .fire().
    fire() is a generator — callers must 'yield from' it to propagate any
    AutopilotStep objects that hooks emit.

    Example
    -------
        outcome = yield from ctx.hooks.fire(
            HookContext(HookEvent.BEFORE_TOOL, agent_name, ctx, tool_name=name, args=args)
        )
        if outcome.decision is Decision.ABORT:
            break
    """

    def __init__(self) -> None:
        self._hooks: list[Hook] = []

    def register(self, hook: Hook) -> "HookManager":
        """Add a hook and keep the list sorted by priority (ascending)."""
        self._hooks.append(hook)
        self._hooks.sort(key=lambda h: h.priority)
        return self

    def fire(
        self, hc: HookContext
    ) -> "Generator[AutopilotStep, None, HookOutcome]":
        """Run all hooks registered for hc.event, merge outcomes, return result.

        Stops early if any hook returns ABORT.
        """
        outcome = HookOutcome.cont()
        for hook in self._hooks:
            if hc.event not in hook.events:
                continue
            try:
                hook_result = yield from hook.handle(hc)
            except Exception as exc:
                log.warning(
                    "HookManager | hook=%s event=%s raised %s — skipping",
                    hook.name, hc.event.value, exc,
                )
                continue
            if hook_result is not None:
                outcome = outcome.merge(hook_result)
            if outcome.decision is Decision.ABORT:
                log.debug(
                    "HookManager | ABORT from hook=%s event=%s reason=%r",
                    hook.name, hc.event.value, outcome.reason,
                )
                break
        return outcome

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def registered_hooks(self) -> list[Hook]:
        """Return the current hook list in priority order (read-only view)."""
        return list(self._hooks)

    def __repr__(self) -> str:
        names = [h.name for h in self._hooks]
        return f"HookManager(hooks={names})"
