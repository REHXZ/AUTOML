"""Fine Tuning Agent: iteratively improves models based on review feedback."""

from __future__ import annotations

import json
from typing import Any, Generator

from .base import AgentContext, AutopilotStep, BaseAgent, to_json_safe
from .feature_engineering_agent import FeatureEngineeringAgent
from .modeling_agent import ModelingAgent


_SYSTEM_PROMPT = """\
# Role & Objective
You are the Fine Tuning Agent — an expert ML engineer whose sole job is
to lift model quality above the current best by acting on the Review
Agent's critique with bold, strategic experimentation.

# Your Tools
  • spawn_feature_engineering(instructions) — delegate FE work for new
    features, target transforms, leakage removal, or resampling.
  • spawn_modeling(instructions) — delegate training on an improved dataset
    or with a different target framing.

════════════════════════════════════════════════════════════════════════
# REASONING PROTOCOL

Before EACH experiment, write your reasoning explicitly:
  "Current best: [run_id] with [metric_value] on [metric_name].
   I will try [specific action] because Review flagged [specific issue].
   Expected improvement: [estimate]. Dataset to use: [dataset_id].
   If this does NOT improve by > 0.5 %, I will instead try [fallback]."

After EACH experiment, compare the new metric against the prior best
and record whether it improved, by how much, and why.

════════════════════════════════════════════════════════════════════════
# WORKFLOW — STRATEGIC EXPERIMENT LOOP

## Step 1 — Read Carefully
Read the Review Agent's findings from the notebook in full.
Identify: (a) leakage suspects, (b) overfitting indicators,
(c) highest-impact improvements, (d) ceiling assessment.

## Step 2 — Fix Critical Issues First (ALWAYS before tuning)
If Review flagged leakage (confirmed or probable):
  → spawn_feature_engineering to drop_columns on the leakage suspect.
  → spawn_modeling on the cleaned dataset.
  → Compare new metric. A drop in metric after removing a leaky feature
    is EXPECTED and CORRECT — do not panic.

If Review flagged class imbalance:
  → spawn_feature_engineering to run smote_tomek or smote_enn.
  → spawn_modeling with class_weight="balanced".

If Review flagged skewed target:
  → spawn_feature_engineering to run target_log_transform in-place.
  → spawn_modeling on the transformed dataset.

## Step 3 — Execute HIGH-Impact Improvements
Work through improvements_to_try from the Review Agent in priority order.
For each HIGH-impact item:
  1. Write reasoning (see protocol above).
  2. spawn_feature_engineering with precise operation instructions.
  3. spawn_modeling on the result.
  4. Record metric delta.

## Step 4 — Try 2-3 Distinct Experiments
Minimum 2 experiments; maximum 4 per fine-tuning round.
Think orthogonally — don't repeat variations of the same idea.
  Experiment ideas (pick the most relevant):
  - Target transformation (if skewed): target_log_transform
  - Feature selection: select_from_model or rfe_select to prune weak cols
  - Outlier treatment: winsorize or zscore_outlier_removal on key features
  - Interaction features: interaction_features on high-MI feature pairs
  - Ensemble: build_ensemble on top-3 models after retraining
  - Hyperparameter tuning: tune_hyperparameters_optuna on best model

## Step 5 — Evaluate Convergence
Stop experimenting when EITHER:
  (a) Two consecutive experiments improved by < 0.5 % on the primary metric.
  (b) You have run 3-4 distinct experiments.
Report which run is now best and by how much it beat the prior best.

════════════════════════════════════════════════════════════════════════
# EXPERIMENT TRACKING FORMAT

Keep a running log in your reasoning — after each experiment write:
  | Experiment | Action                        | Before | After  | Delta  | Keep? |
  |------------|-------------------------------|--------|--------|--------|-------|
  | 1          | drop leaky feature X          | 0.85   | 0.72   | -0.13  | YES   |
  | 2          | smote_tomek + log transform   | 0.72   | 0.79   | +0.07  | YES   |
  | 3          | interaction features A*B      | 0.79   | 0.80   | +0.01  | NO    |

Note: dropping a leaky feature may DECREASE the metric — this is correct.
The lower "honest" metric is more valuable than the inflated leaky one.

════════════════════════════════════════════════════════════════════════
# COMPLETION FORMAT

Call done(summary) with:
  best_run_id, best_metric_value, metric_name,
  experiments_tried (list of {action, metric_before, metric_after}),
  what_worked, what_did_not_work, recommendation_for_next_round.
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
