"""Review Agent: critiques training runs and proposes improvements."""

from __future__ import annotations

import json
from typing import Any, Generator

from .base import AgentContext, AutopilotStep, BaseAgent, to_json_safe


_SYSTEM_PROMPT = """\
You are the Review Agent — a sceptical ML critic whose JOB is to drive
metrics UP (accuracy / F1 / R²) and error DOWN (RMSE / MAE / log-loss).

You will see one or more training runs in the notebook. For EACH run:

1. AUDIT THE METRICS.
   • Compare against the baseline (DummyClassifier majority class, or
     DummyRegressor mean). If the best model only matches the baseline,
     the model has no signal — call this out.
   • Flag suspiciously perfect scores (R² ≥ 0.99, F1 ≥ 0.99, accuracy
     ≥ 0.99) as probable LEAKAGE. Name the columns you suspect.
   • Check the gap between the best model and the second-best. A 10×
     gap usually points at leakage in the winner.
   • For classification, watch for class imbalance — high accuracy with
     low recall on the minority class is misleading.

2. RANK CONCRETE IMPROVEMENTS by expected impact. Use ONLY operations
   the Feature Engineering Agent supports:
     • groupby_aggregate  – materialise summary targets (e.g. monthly
       order count) when the raw target is too sparse / noisy.
     • drop_high_missing  – cull cols with > 50 % missing.
     • encode_dates       – year/month/day/dayofweek/quarter features.
     • log_transform      – log1p numeric features with long tails.
     • target_log_transform – log1p a right-skewed regression target.
     • one_hot_encode     – low-cardinality categoricals.
     • bin_numeric        – quantile bin a noisy continuous feature.
     • interaction_features – multiply two numeric cols when joint
       signal is suspected.
     • polynomial_features – squared/cubed terms for non-linearity.
     • filter_outliers    – IQR trim before fitting linear models.
     • drop_columns       – remove a leakage-suspect column and retrain.
   The Modeling Agent can also vary test_size or random_state.

3. FORMAT YOUR RECOMMENDATIONS. For each item in improvements_to_try
   write a single line in the form:
     "[expected_impact: HIGH|MED|LOW] <FE operation or model action>
      on dataset <id> → <expected metric delta, e.g. +0.05 R²>"
   Order the array from HIGHEST expected impact downward — the Fine
   Tuning Agent will execute them in order.

4. SAY THE QUIET PART OUT LOUD. If the headline metric is already at
   ceiling for the data quality, say so honestly so the Scientist
   doesn't waste rounds chasing a noisy +0.001.

Call record_finding(text) for each major critique, then done(summary)
with the structured recommendations.
"""


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "record_finding",
                "description": "Write a critique to the shared notebook.",
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
                "description": "Finish review with structured recommendations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "trust_score": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "How much to trust the reported metrics.",
                        },
                        "leakage_suspected": {"type": "boolean"},
                        "best_run_id": {"type": "string"},
                        "issues": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "improvements_to_try": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Concrete next experiments (feature ideas, target reframings, etc.)",
                        },
                        "narrative": {"type": "string"},
                    },
                    "required": ["narrative"],
                },
            },
        },
    ]


class ReviewAgent(BaseAgent):
    name = "review"
    display_name = "Review Agent"

    def __init__(self, client, deployment: str, context: AgentContext) -> None:
        super().__init__(client, deployment, context)
        self._summary: dict[str, Any] = {}

    def run(
        self, instructions: str
    ) -> Generator[AutopilotStep, list[str] | None, dict[str, Any]]:
        yield self._step(
            "agent_start",
            "Review Agent dispatched",
            instructions or "(critique the runs and propose improvements)",
        )

        user_prompt = (
            f"Scientist's instructions:\n{instructions}\n\n"
            f"Training runs to review:\n{self._ctx.training_runs_summary()}\n\n"
            f"Notebook so far:\n{self._ctx.notebook_text()}\n\n"
            f"User goal: {self._ctx.user_goal or '(none)'}\n\n"
            "Critique and recommend."
        )

        yield from self.run_llm_loop(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=_tools(),
            dispatch=self._dispatch,
            max_iterations=12,
            thought_title="Review Agent — Reasoning",
        )

        if self._summary:
            yield self._step(
                "review",
                "Review Summary",
                self._summary.get("narrative", ""),
                data=self._summary,
            )

        yield self._step("agent_end", "Review Agent finished", "")
        return self._summary or {"narrative": "Review agent ended without summary."}

    def _dispatch(
        self, name: str, args: dict, tool_call_id: str
    ) -> tuple[str | None, AutopilotStep | None, bool]:
        if name == "record_finding":
            text = (args.get("text") or "").strip()
            if text:
                self._ctx.notebook.append(f"[Review] {text}")
            return (
                json.dumps({"recorded": True}),
                self._step("observation", "Review critique", text),
                False,
            )
        if name == "done":
            self._summary = to_json_safe(args)
            return json.dumps({"status": "noted"}), None, True
        return json.dumps({"error": f"Unknown tool: {name}"}), None, False
