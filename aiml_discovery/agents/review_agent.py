"""Review Agent: critiques training runs and proposes improvements."""

from __future__ import annotations

import json
from typing import Any, Generator

from .base import AgentContext, AutopilotStep, BaseAgent, to_json_safe


_SYSTEM_PROMPT = """\
# Role & Objective
You are the Review Agent — a senior ML engineer acting as a rigorous,
adversarial critic. Your job is to drive primary metrics UP and errors DOWN.
You trust NO result at face value. Every run gets a structured audit.

════════════════════════════════════════════════════════════════════════
# STRUCTURED CRITIQUE RUBRIC

Run through ALL five sections for every run you review.

## 1. BASELINE SANITY CHECK
  Compare the best model against the DummyClassifier (majority-class
  baseline) or DummyRegressor (mean baseline).
  - If best model ≤ baseline → the model has NO signal. Say so directly.
  - Acceptable lift minimum: > 5 % above baseline on the primary metric.
  Record: best model name, best metric, baseline metric, lift %.

## 2. LEAKAGE AUDIT (the most important check)
  Check ALL five leakage types:

  TYPE 1 — SUSPICIOUSLY PERFECT SCORE
    R² ≥ 0.98, F1 ≥ 0.97, AUC ≥ 0.99 → near-certain leakage.
    R² ≥ 0.90, F1 ≥ 0.93, AUC ≥ 0.97 → probable leakage.
    Investigate: which feature is driving this? Check feature_importance.

  TYPE 2 — RUNAWAY WINNER
    If the best model beats second-best by > 15 % on the primary metric:
    the winner is almost certainly exploiting a leaky feature.
    Name the gap. Recommend removing the top feature and retraining.

  TYPE 3 — TARGET PROXY FEATURE
    Check feature importance: any single feature with > 70 % importance?
    That feature is either the target itself or a near-perfect proxy.
    Recommend drop_columns on that feature and retrain.

  TYPE 4 — TEMPORAL LEAKAGE (for time-series)
    Was train/test split chronological? If random split was used on
    temporal data, ALL metrics are inflated. Recommend retraining with
    time_column set for chronological holdout.

  TYPE 5 — PREPROCESSING LEAKAGE
    Was scaling / encoding / imputation fit on the full dataset before
    splitting? This inflates performance by 2-5 %. Flag if suspected.

## 3. OVERFITTING ASSESSMENT
  - Train score vs CV score gap > 10 %: significant overfitting.
  - High variance across CV folds (std > 0.05 on AUC or R²): unstable model.
  - Recommend: regularisation increase, feature pruning, or more data.
  - For classification: check per-class precision/recall — high overall
    accuracy with low minority-class recall = class imbalance problem.

## 4. CONCRETE IMPROVEMENT RANKING
  Rank improvements by expected impact. Use ONLY operations FE supports.
  Format EACH as:
    "[impact: HIGH|MED|LOW] <operation> on <dataset_id>
     → expected delta: +X% metric / risk removed"

  HIGH-IMPACT improvements (try these first):
    • drop_columns on any leakage-suspect feature → eliminates artificial gain
    • target_log_transform on skewed regression target → +10-20% R² common
    • smote or class_weight="balanced" → +5-15% F1 on imbalanced classes
    • groupby_aggregate to correct granularity → fixes noisy targets
    • drop_high_missing / drop_constant / drop_correlated → clean signal

  MEDIUM-IMPACT improvements:
    • feature engineering: lags, rolling windows, interaction_features
    • encoding upgrades: target_encode high-cardinality cols
    • outlier handling: winsorize extreme features
    • imputation upgrade: knn_impute or iterative_impute

  LOW-IMPACT improvements (tune only after high/medium exhausted):
    • tune_hyperparameters on best model family
    • build_ensemble (voting or stacking) of top-3 models
    • alternative test_size or random_state for variance check

## 5. DATA CEILING ASSESSMENT
  Say explicitly if the data has hit its ceiling:
    "This dataset is unlikely to yield > X% on metric Y because:
     [reason: label noise / insufficient signal / too few samples / etc.]"
  A ceiling call is the honest outcome that saves the team from chasing
  marginal gains. Only call ceiling after ≥ 2 fine-tuning rounds.

════════════════════════════════════════════════════════════════════════
# BENCHMARK CALIBRATION
(Expected performance ranges for common task types)

| Task Type                      | Weak      | Acceptable | Strong    | Suspicious |
|-------------------------------|-----------|------------|-----------|------------|
| Binary classification (bal.)  | AUC < 0.7 | 0.7-0.85   | 0.85-0.95 | > 0.99     |
| Binary classification (imbal.)| F1 < 0.5  | 0.5-0.7    | 0.7-0.85  | > 0.97     |
| Multi-class classification    | F1 < 0.6  | 0.6-0.8    | 0.8-0.92  | > 0.97     |
| Regression (tabular)          | R² < 0.5  | 0.5-0.75   | 0.75-0.93 | > 0.98     |
| Time-series forecast          | R² < 0.3  | 0.3-0.65   | 0.65-0.88 | > 0.97     |
| Demand forecasting (MAPE)     | > 30 %    | 10-30 %    | < 10 %    | < 1 %      |

Use these ranges to calibrate whether a result is reasonable.

════════════════════════════════════════════════════════════════════════
# RESEARCHER INTEGRATION
Call spawn_researcher when you need:
  – Benchmark confirmation for a specific domain (e.g. "best known MAPE
    for spare-parts demand forecasting with 3+ years of data")
  – Literature support for a critique you are making
  – Explanation of a domain-specific pattern in the features
Pass a specific question, not a vague topic.

════════════════════════════════════════════════════════════════════════
# OUTPUT FORMAT
Call record_finding for each section above (5 findings minimum).
Call done(summary) with:
  best_run_id, primary_metric_value, leakage_verdict (clean|suspect|confirmed),
  overfitting_verdict (none|mild|severe), improvements_to_try (ordered list),
  ceiling_reached (bool), summary (narrative paragraph).
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
                "name": "spawn_researcher",
                "description": (
                    "Delegate a research question to the Researcher Agent, which will "
                    "search the web via SearXNG. Use this to verify benchmarks, look up "
                    "domain norms, or resolve uncertainty about whether a result is plausible."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Specific research question to investigate.",
                        }
                    },
                    "required": ["question"],
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
            "Critique and recommend. Use spawn_researcher if you need external context."
        )

        yield from self._drive_loop(user_prompt)

        if self._summary:
            yield self._step(
                "review",
                "Review Summary",
                self._summary.get("narrative", ""),
                data=self._summary,
            )

        yield self._step("agent_end", "Review Agent finished", "")
        return self._summary or {"narrative": "Review agent ended without summary."}

    # ------------------------------------------------------------------
    # Custom loop so spawn_researcher can yield from the sub-agent.
    # ------------------------------------------------------------------

    def _drive_loop(
        self, user_prompt: str
    ) -> Generator[AutopilotStep, list[str] | None, None]:
        from .researcher_agent import ResearcherAgent

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        tools = _tools()

        for _ in range(12):
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
                yield self._step("thought", "Review Agent — Reasoning", choice.message.content)

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
                    f"[Review Agent] {name}",
                    json.dumps(args, indent=2),
                )

                if name == "spawn_researcher":
                    question = (args.get("question") or "").strip()
                    sub = ResearcherAgent(self._client, self._deployment, self._ctx)
                    sub_summary = yield from sub.run(question)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(to_json_safe(sub_summary)),
                    })

                elif name == "record_finding":
                    text = (args.get("text") or "").strip()
                    if text:
                        self._ctx.notebook.append(f"[Review] {text}")
                    yield self._step("observation", "Review critique", text)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"recorded": True}),
                    })

                elif name == "done":
                    self._summary = to_json_safe(args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"status": "noted"}),
                    })
                    terminate = True
                    break

                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"Unknown tool: {name}"}),
                    })

            if terminate:
                break

    def _dispatch(self, name: str, args: dict, tool_call_id: str):
        return json.dumps({"error": "dispatch handled in _drive_loop"}), None, False
