"""Shared agent infrastructure: types, context, base class, and utilities."""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generator

if TYPE_CHECKING:
    from ..session_store import SessionWriter

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..logging_setup import configure_logging
from ..storage import DatasetInfo, ProjectStore

configure_logging()
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Shared step type
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AutopilotStep:
    """A unit of activity yielded from any agent to the UI."""

    index: int
    # "thought" | "tool_call" | "tool_result" | "chart" | "ask" |
    # "new_dataset" | "training" | "summary" | "observation" |
    # "agent_start" | "agent_end" | "review"
    kind: str
    title: str
    detail: str = ""
    data: dict[str, Any] | None = None
    agent: str = "scientist"


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
    _step_counter: int = 0

    def next_step_index(self) -> int:
        self._step_counter += 1
        return self._step_counter

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
# Azure OpenAI client factory
# ──────────────────────────────────────────────────────────────────────────────


def build_azure_client(api_key: str):
    from openai import AzureOpenAI

    api_base = os.environ.get("OPENAI_API_BASE", "")
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=api_base,
        api_version="2024-12-01-preview",
    )


def get_deployment() -> str:
    return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")


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
        return AutopilotStep(
            index=self._ctx.next_step_index(),
            kind=kind,
            title=title,
            detail=detail,
            data=data,
            agent=self.name,
        )

    def _persist_message(self, message: dict) -> None:
        session = getattr(self._ctx, "session", None)
        if session is not None:
            session.append_message(message)

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
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self._persist_message(messages[0])
        self._persist_message(messages[1])
        thought_title = thought_title or f"{self.display_name} — Reasoning"

        for iteration in range(max_iterations):
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
            choice = response.choices[0]
            log.debug(
                "LLM response | agent=%s finish_reason=%s has_tool_calls=%s",
                self.name, choice.finish_reason, bool(choice.message.tool_calls),
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

                log.info(
                    "tool_call | agent=%s name=%s args=%s",
                    self.name, name, raw_args[:600],
                )

                yield self._step(
                    "tool_call",
                    f"[{self.display_name}] {name}",
                    json.dumps(args, indent=2),
                )

                tool_content, extra_step, terminate = dispatch(name, args, tc.id)

                if extra_step is not None:
                    yield extra_step

                if tool_content is None:
                    log.warning(
                        "tool_result | agent=%s name=%s returned None content",
                        self.name, name,
                    )
                elif isinstance(tool_content, list):
                    log.debug(
                        "tool_result | agent=%s name=%s content=vision+text parts=%d",
                        self.name, name, len(tool_content),
                    )
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    }
                    messages.append(tool_msg)
                    self._persist_message(tool_msg)
                else:
                    log.debug(
                        "tool_result | agent=%s name=%s content_len=%d",
                        self.name, name, len(tool_content),
                    )
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
