"""Fine Tuning Agent: iteratively improves models based on review feedback."""

from __future__ import annotations

import json
from typing import Any, Generator

from .base import AgentContext, AutopilotStep, BaseAgent, to_json_safe
from .feature_engineering_agent import FeatureEngineeringAgent
from .modeling_agent import ModelingAgent


_SYSTEM_PROMPT = """\
You are the Fine Tuning Agent — your job is to lift model quality by acting
on the Review Agent's critique.

You have two sub-tools at your disposal:
  • spawn_feature_engineering(instructions) — delegate feature work to the
    Feature Engineering Agent. Use this when the fix requires building new
    features, transforming the target, or removing leakage columns.
  • spawn_modeling(instructions) — delegate training to the Modeling Agent.
    Use this to retrain on a new dataset, a new target framing, or a
    different test_size / random_state.

Workflow:
  1. Read the review findings and the notebook carefully.
  2. Pick the highest-impact improvement and build any needed features.
  3. Retrain on the improved dataset.
  4. Repeat with the next-best idea — try 2-3 distinct experiments.
  5. Call done(summary) with a JSON of what you tried, which worked, and
     which run is now the best.

Be bold and curious. Try variants the Review Agent did not explicitly call
out if you think they might help. Compare new metrics against the previous
best honestly.
"""


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "spawn_feature_engineering",
                "description": "Delegate to the Feature Engineering Agent. Give detailed instructions.",
                "parameters": {
                    "type": "object",
                    "properties": {"instructions": {"type": "string"}},
                    "required": ["instructions"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "spawn_modeling",
                "description": "Delegate to the Modeling Agent. Specify dataset_id(s) and target_column(s).",
                "parameters": {
                    "type": "object",
                    "properties": {"instructions": {"type": "string"}},
                    "required": ["instructions"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_finding",
                "description": "Write a tuning note to the shared notebook.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "done",
                "description": "Finish tuning with a summary of experiments.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "experiments": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "best_run_id": {"type": "string"},
                        "improvement_delta": {"type": "string"},
                        "narrative": {"type": "string"},
                    },
                    "required": ["narrative"],
                },
            },
        },
    ]


class FineTuningAgent(BaseAgent):
    name = "fine_tuning"
    display_name = "Fine Tuning Agent"

    def __init__(self, client, deployment: str, context: AgentContext) -> None:
        super().__init__(client, deployment, context)
        self._summary: dict[str, Any] = {}
        # Pending child-agent generator the parent loop will drive.
        self._pending_child: Generator[AutopilotStep, list[str] | None, dict[str, Any]] | None = None

    def run(
        self, instructions: str
    ) -> Generator[AutopilotStep, list[str] | None, dict[str, Any]]:
        yield self._step(
            "agent_start",
            "Fine Tuning Agent dispatched",
            instructions or "(iterate on review feedback)",
        )

        user_prompt = (
            f"Scientist's instructions:\n{instructions}\n\n"
            f"Notebook so far:\n{self._ctx.notebook_text()}\n\n"
            f"Existing training runs:\n{self._ctx.training_runs_summary()}\n\n"
            "Choose improvements to try and execute them via your sub-tools."
        )

        # We cannot yield from within a sync callback, so we drive the loop
        # manually here and forward sub-agent yields ourselves.
        yield from self._drive_loop(user_prompt)

        yield self._step("agent_end", "Fine Tuning Agent finished", "")
        return self._summary or {"narrative": "Fine tuning ended without summary."}

    # ------------------------------------------------------------------
    # Custom loop so we can `yield from` sub-agents from inside dispatch.
    # ------------------------------------------------------------------

    def _drive_loop(
        self, user_prompt: str
    ) -> Generator[AutopilotStep, list[str] | None, None]:
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        tools = _tools()

        for _ in range(20):
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            choice = response.choices[0]

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": choice.message.content,
            }
            if choice.message.tool_calls:
                assistant_msg["tool_calls"] = [
                    tc.model_dump() for tc in choice.message.tool_calls
                ]
            messages.append(assistant_msg)

            if choice.message.content:
                yield self._step("thought", "Fine Tuning — Reasoning", choice.message.content)

            if choice.finish_reason == "stop":
                break
            if choice.finish_reason != "tool_calls":
                continue

            terminate = False
            for tc in choice.message.tool_calls:
                name = tc.function.name
                args: dict[str, Any] = json.loads(tc.function.arguments or "{}")

                yield self._step(
                    "tool_call",
                    f"[Fine Tuning] {name}",
                    json.dumps(args, indent=2),
                )

                if name == "spawn_feature_engineering":
                    sub = FeatureEngineeringAgent(self._client, self._deployment, self._ctx)
                    sub_summary = yield from sub.run(args.get("instructions", ""))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(to_json_safe(sub_summary)),
                        }
                    )
                elif name == "spawn_modeling":
                    sub_m = ModelingAgent(self._client, self._deployment, self._ctx)
                    sub_summary = yield from sub_m.run(args.get("instructions", ""))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(to_json_safe(sub_summary)),
                        }
                    )
                elif name == "record_finding":
                    text = (args.get("text") or "").strip()
                    if text:
                        self._ctx.notebook.append(f"[Tuning] {text}")
                        yield self._step("observation", "Tuning note", text)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"recorded": True}),
                        }
                    )
                elif name == "done":
                    self._summary = to_json_safe(args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"status": "noted"}),
                        }
                    )
                    terminate = True
                    break
                else:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"error": f"Unknown tool: {name}"}),
                        }
                    )

            if terminate:
                break

    # Unused stub so BaseAgent's interface stays consistent.
    def _dispatch(self, name: str, args: dict, tool_call_id: str):
        return json.dumps({"error": "dispatch handled in _drive_loop"}), None, False
