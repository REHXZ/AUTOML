"""Shared agent infrastructure: types, context, base class, and utilities."""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generator

if TYPE_CHECKING:
    from backend.services.session_store import SessionWriter
    from .hooks import HookContext, HookManager, HookOutcome

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from backend.server.logging_setup import configure_logging
from backend.services.project_store import DatasetInfo, ProjectStore

configure_logging()
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Shared step type
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# AIML lifecycle (modified CRISP-DM, no deployment, with iteration loop).
# Soft guidance only — phases are tracked as metadata on every step so the
# exported notebook can group activity into lifecycle sections, but the
# Scientist is not gated on phase transitions.
# ──────────────────────────────────────────────────────────────────────────────

PHASES: list[dict[str, str]] = [
    {
        "id": "business_understanding",
        "title": "Business Understanding",
        "description": (
            "Frame the problem with the user: clarify the target, the unit of "
            "prediction, the success metric, and any business constraints."
        ),
    },
    {
        "id": "data_understanding",
        "title": "Data Understanding",
        "description": (
            "Profile every available dataset. Inspect distributions, missingness, "
            "cardinality, and target behaviour. Output: a clear picture of what "
            "the data contains and which signals look promising."
        ),
    },
    {
        "id": "data_preparation",
        "title": "Data Preparation",
        "description": (
            "Build modelling-ready datasets: clean, transform, aggregate, and "
            "engineer features. For forecasting, this is where lead targets and "
            "lag/rolling features get materialised."
        ),
    },
    {
        "id": "modeling",
        "title": "Modeling",
        "description": (
            "Train baseline models and run fine-tuning rounds. Each fine-tuning "
            "pass tries the improvements the Review Agent recommended."
        ),
    },
    {
        "id": "evaluation",
        "title": "Evaluation",
        "description": (
            "Compare runs against each other and against the success metric. "
            "Critique each run for leakage, drift, and over-fitting."
        ),
    },
    {
        "id": "iteration",
        "title": "Iteration & User Feedback",
        "description": (
            "Decide what to do next: loop back to modeling (more tuning), back "
            "to data preparation (new features), or finalise. Capture user "
            "feedback on the chosen direction."
        ),
    },
]

PHASE_IDS: list[str] = [p["id"] for p in PHASES]
PHASE_BY_ID: dict[str, dict[str, str]] = {p["id"]: p for p in PHASES}


@dataclass
class AutopilotStep:
    """A unit of activity yielded from any agent to the UI."""

    index: int
    # "thought" | "tool_call" | "tool_result" | "chart" | "ask" |
    # "new_dataset" | "training" | "summary" | "observation" |
    # "agent_start" | "agent_end" | "review" | "phase_transition"
    kind: str
    title: str
    detail: str = ""
    data: dict[str, Any] | None = None
    agent: str = "scientist"
    # Which lifecycle phase this step belongs to. Auto-populated from
    # AgentContext.current_phase by BaseAgent._step().
    phase: str = "business_understanding"


# ──────────────────────────────────────────────────────────────────────────────
# Shared mutable context across all agents
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AgentContext:
    project_id: str
    store: ProjectStore
    user_goal: str = ""
    # Free-form observations from any agent. The Scientist consults this to
    # decide next moves.
    notebook: list[str] = field(default_factory=list)
    # Outputs accumulated as the run progresses.
    new_datasets: list[DatasetInfo] = field(default_factory=list)
    training_runs: list[dict[str, Any]] = field(default_factory=list)
    # Answers the Scientist has already collected from the user.
    user_answers: list[dict[str, str]] = field(default_factory=list)
    # Optional persistence handle. When set, agents stream their LLM messages
    # to disk so the run can be resumed after a refresh.
    session: "SessionWriter | None" = None
    # Current AIML lifecycle phase. Soft guidance — the Scientist updates this
    # via set_phase() but agents can still operate freely. Every step yielded
    # via BaseAgent._step() is tagged with this value so the exported notebook
    # can group activity into lifecycle sections.
    current_phase: str = "business_understanding"
    # Set True by the stop endpoint so loops exit before starting the next
    # LLM API call instead of waiting for the blocking call to complete.
    should_stop: bool = False
    _step_counter: int = 0
    # Per-agent token usage accumulated during the run.
    # {agent_name: {"prompt_tokens": int, "completion_tokens": int, "calls": int,
    #               "last_prompt_tokens": int}}
    agent_token_usage: dict = field(default_factory=dict)
    # ── Hook lifecycle (set by AiAutopilot before any agent runs) ──
    # Optional hook manager shared by all agents in a run.  When None every
    # _fire() call is a no-op so the system behaves exactly as before.
    hooks: "HookManager | None" = None
    # OpenAI client + deployment stored here so that hook policies can spawn
    # sub-agents (e.g. ModelTesterGateHook) without holding a back-reference.
    client: Any = None
    deployment: str = ""
    # Run IDs already evaluated by ModelTesterAgent; used by the gate hook to
    # avoid re-running the tester (previously tracked on AimlScientist instance).
    tested_run_ids: set = field(default_factory=set)

    def next_step_index(self) -> int:
        self._step_counter += 1
        return self._step_counter

    def set_phase(self, phase: str) -> str:
        """Update current_phase if valid; return the resulting phase id."""
        from .base import PHASE_IDS  # local import keeps module bootstrap clean
        if phase in PHASE_IDS:
            self.current_phase = phase
        return self.current_phase

    def list_datasets(self) -> list[DatasetInfo]:
        return self.store.list_datasets(self.project_id)

    def find_dataset(self, dataset_id: str) -> DatasetInfo | None:
        return next(
            (d for d in self.list_datasets() if d.id == dataset_id),
            None,
        )

    def notebook_text(self) -> str:
        if not self.notebook:
            return "(empty)"
        return "\n".join(f"- {entry}" for entry in self.notebook)

    def training_runs_summary(self) -> str:
        if not self.training_runs:
            return "(no training runs yet)"
        lines = []
        for r in self.training_runs:
            metrics = ", ".join(
                f"{k}={v:.4f}" for k, v in r.get("best_metrics", {}).items()
            )
            lines.append(
                f"- run_id={r.get('run_id')} dataset={r.get('dataset')} "
                f"target={r.get('target')} task={r.get('task_type')} "
                f"model={r.get('best_model')} metrics=({metrics})"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI client factory — uses Azure if OPENAI_API_BASE is set, else standard OpenAI
# ──────────────────────────────────────────────────────────────────────────────


def build_azure_client(api_key: str):
    api_base = os.environ.get("OPENAI_API_BASE", "").strip()
    if api_base:
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=api_base,
            api_version="2024-12-01-preview",
        )
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def get_deployment() -> str:
    api_base = os.environ.get("OPENAI_API_BASE", "").strip()
    if api_base:
        return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
    return os.environ.get("OPENAI_MODEL", "gpt-5.4")


# ──────────────────────────────────────────────────────────────────────────────
# Base agent
# ──────────────────────────────────────────────────────────────────────────────


# Type alias for the tool dispatcher each agent provides.
# Returns: (tool_content_for_llm, optional_extra_step, terminate_flag)
ToolDispatcher = Callable[
    [str, dict, str],
    tuple[str | list | None, AutopilotStep | None, bool],
]


class BaseAgent:
    """Common scaffolding shared by all LLM-driven agents."""

    name: str = "base"
    display_name: str = "Base Agent"

    def __init__(self, client, deployment: str, context: AgentContext) -> None:
        self._client = client
        self._deployment = deployment
        self._ctx = context

    # ------------------------------------------------------------------
    # Step factories
    # ------------------------------------------------------------------

    def _step(
        self,
        kind: str,
        title: str,
        detail: str = "",
        data: dict | None = None,
    ) -> AutopilotStep:
        # Inject current context size so every step carries the agent's
        # prompt-token count.  Reads from shared ctx so TokenAccountingHook
        # (which writes there) is the single source of truth.
        d: dict = dict(data) if data else {}
        last_tokens = self._ctx.agent_token_usage.get(self.name, {}).get(
            "last_prompt_tokens"
        )
        if last_tokens is not None:
            d["context_tokens"] = last_tokens
        return AutopilotStep(
            index=self._ctx.next_step_index(),
            kind=kind,
            title=title,
            detail=detail,
            data=d if d else None,
            agent=self.name,
            phase=self._ctx.current_phase,
        )

    def _persist_message(self, message: dict) -> None:
        session = getattr(self._ctx, "session", None)
        if session is not None:
            session.append_message(message)

    # ------------------------------------------------------------------
    # Hook helpers
    # ------------------------------------------------------------------

    def _fire(
        self, hc: "HookContext"
    ) -> "Generator[AutopilotStep, None, HookOutcome]":
        """Fire a hook event.  No-op (returns CONTINUE) when no manager is set."""
        from .hooks import HookOutcome
        if self._ctx.hooks is None:
            yield from ()
            return HookOutcome.cont()
        return (yield from self._ctx.hooks.fire(hc))

    def _invoke_llm(
        self,
        messages: list[dict],
        tools: list[dict],
        iteration: int,
    ) -> "Generator[AutopilotStep, None, tuple[Any, HookOutcome]]":
        """Fire BEFORE_LLM, call the API, fire AFTER_LLM.

        Returns (choice, outcome).  When BEFORE_LLM returns ABORT the API call
        is skipped and choice is None.
        """
        from .hooks import HookContext, HookEvent, HookOutcome

        hc_before = HookContext(
            event=HookEvent.BEFORE_LLM,
            agent_name=self.name,
            ctx=self._ctx,
            messages=messages,
            iteration=iteration,
        )
        pre = yield from self._fire(hc_before)
        if pre.decision.value == "abort":
            return None, pre

        log.debug(
            "LLM call | agent=%s model=%s iteration=%d messages=%d",
            self.name, self._deployment, iteration, len(messages),
        )
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        hc_after = HookContext(
            event=HookEvent.AFTER_LLM,
            agent_name=self.name,
            ctx=self._ctx,
            messages=messages,
            iteration=iteration,
            response=response,
        )
        yield from self._fire(hc_after)

        return response.choices[0], HookOutcome.cont()

    # ------------------------------------------------------------------
    # LLM tool-calling loop
    # ------------------------------------------------------------------

    def run_llm_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict],
        dispatch: ToolDispatcher,
        max_iterations: int = 40,
        thought_title: str | None = None,
    ) -> Generator[AutopilotStep, list[str] | None, list[dict]]:
        """Drive an OpenAI tool-calling conversation.

        Yields AutopilotStep objects for each thought/tool_call/tool_result and
        returns the full message history for downstream inspection.
        """
        from .hooks import Decision, HookContext, HookEvent

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self._persist_message(messages[0])
        self._persist_message(messages[1])
        thought_title = thought_title or f"{self.display_name} — Reasoning"

        for iteration in range(max_iterations):
            choice, pre_outcome = yield from self._invoke_llm(messages, tools, iteration)
            if choice is None or pre_outcome.decision is Decision.ABORT:
                log.info(
                    "LLM loop aborting | agent=%s iteration=%d reason=%r",
                    self.name, iteration, pre_outcome.reason,
                )
                break

            last_tokens = self._ctx.agent_token_usage.get(self.name, {}).get("last_prompt_tokens")
            log.debug(
                "LLM response | agent=%s finish_reason=%s has_tool_calls=%s context_tokens=%s",
                self.name, choice.finish_reason, bool(choice.message.tool_calls),
                last_tokens,
            )

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": choice.message.content,
            }
            if choice.message.tool_calls:
                assistant_msg["tool_calls"] = [
                    tc.model_dump() for tc in choice.message.tool_calls
                ]
            messages.append(assistant_msg)
            self._persist_message(assistant_msg)

            if choice.message.content:
                yield self._step("thought", thought_title, choice.message.content)

            if choice.finish_reason == "stop":
                log.info("LLM loop finished | agent=%s iterations=%d", self.name, iteration + 1)
                break

            if choice.finish_reason != "tool_calls":
                log.warning(
                    "Unexpected finish_reason=%r | agent=%s — skipping iteration",
                    choice.finish_reason, self.name,
                )
                continue

            terminate_outer = False
            for tc in choice.message.tool_calls:
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                args: dict[str, Any] = json.loads(raw_args)

                # ── BEFORE_TOOL hook ────────────────────────────────────
                hc_before = HookContext(
                    event=HookEvent.BEFORE_TOOL,
                    agent_name=self.name,
                    ctx=self._ctx,
                    tool_name=name,
                    args=args,
                    extra={"tool_call_id": tc.id},
                )
                pre_tool = yield from self._fire(hc_before)

                if pre_tool.decision is Decision.ABORT:
                    log.info(
                        "tool_call ABORTED by hook | agent=%s name=%s reason=%r",
                        self.name, name, pre_tool.reason,
                    )
                    terminate_outer = True
                    break

                if pre_tool.decision is Decision.MODIFY and pre_tool.args is not None:
                    args = pre_tool.args

                yield self._step(
                    "tool_call",
                    f"[{self.display_name}] {name}",
                    json.dumps(args, indent=2),
                )

                # ── Dispatch (with SKIP short-circuit) ─────────────────
                if pre_tool.decision is Decision.SKIP:
                    log.info(
                        "tool_call SKIPPED by hook | agent=%s name=%s reason=%r",
                        self.name, name, pre_tool.reason,
                    )
                    tool_content = pre_tool.result if pre_tool.result is not None else json.dumps({"skipped": True})
                    extra_step = None
                    terminate = False
                else:
                    try:
                        tool_content, extra_step, terminate = dispatch(name, args, tc.id)
                    except Exception as exc:
                        hc_err = HookContext(
                            event=HookEvent.TOOL_ERROR,
                            agent_name=self.name,
                            ctx=self._ctx,
                            tool_name=name,
                            args=args,
                            error=exc,
                        )
                        err_out = yield from self._fire(hc_err)
                        if err_out.decision is Decision.SKIP:
                            tool_content = err_out.result if err_out.result is not None else json.dumps({"error": str(exc)})
                            extra_step = None
                            terminate = False
                        else:
                            raise

                if extra_step is not None:
                    yield extra_step

                # ── AFTER_TOOL hook ─────────────────────────────────────
                hc_after = HookContext(
                    event=HookEvent.AFTER_TOOL,
                    agent_name=self.name,
                    ctx=self._ctx,
                    tool_name=name,
                    args=args,
                    result=tool_content,
                )
                post_tool = yield from self._fire(hc_after)
                if post_tool.decision is Decision.MODIFY and post_tool.result is not None:
                    tool_content = post_tool.result
                if post_tool.decision is Decision.ABORT:
                    terminate = True

                if tool_content is None:
                    log.warning(
                        "tool_result | agent=%s name=%s returned None content",
                        self.name, name,
                    )
                elif isinstance(tool_content, list):
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    }
                    messages.append(tool_msg)
                    self._persist_message(tool_msg)
                else:
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    }
                    messages.append(tool_msg)
                    self._persist_message(tool_msg)

                if terminate:
                    terminate_outer = True
                    break

            if terminate_outer:
                break

        return messages


# ──────────────────────────────────────────────────────────────────────────────
# JSON / image utilities
# ──────────────────────────────────────────────────────────────────────────────


def to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serialisable natives."""
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (v != v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def fig_to_base64(fig: go.Figure) -> str | None:
    """Render figure to base64 PNG for LLM vision feedback. None on failure."""
    try:
        import plotly.io as pio

        img_bytes = pio.to_image(fig, format="png", width=900, height=480, scale=1)
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        log.debug("fig_to_base64 | OK png_bytes=%d b64_chars=%d", len(img_bytes), len(b64))
        return b64
    except Exception as exc:
        log.warning("fig_to_base64 | FAILED: %s", exc)
        return None


def vision_tool_content(text_result: str, fig: go.Figure | None) -> str | list:
    """Build OpenAI tool-result content with optional inline chart image."""
    if fig is None:
        log.debug("vision_tool_content | no figure — returning text only")
        return text_result
    b64 = fig_to_base64(fig)
    if not b64:
        log.warning("vision_tool_content | base64 conversion failed — returning text only")
        return text_result
    log.debug("vision_tool_content | returning vision+text content")
    return [
        {"type": "text", "text": text_result},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        },
    ]
