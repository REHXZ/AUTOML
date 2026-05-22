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

5. USE THE RESEARCHER WHEN YOU NEED EXTERNAL CONTEXT.
   Call spawn_researcher if you need to:
     – Verify whether a metric level is reasonable for the domain.
     – Look up known benchmarks for a technique.
     – Understand if a feature pattern is domain-standard or suspicious.
   Pass a specific, focused question — not a vague topic.

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
