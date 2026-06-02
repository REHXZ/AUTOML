"""AIML Scientist Agent: orchestrates EDA, Feature Engineering, Modeling, Review, Fine Tuning."""

from __future__ import annotations

import json
import logging
from typing import Any, Generator

from ..logging_setup import configure_logging
from .base import (
    PHASE_BY_ID,
    PHASE_IDS,
    PHASES,
    AgentContext,
    AutopilotStep,
    BaseAgent,
    to_json_safe,
)

configure_logging()
log = logging.getLogger(__name__)
from .eda_agent import EdaAgent
from .feature_engineering_agent import FeatureEngineeringAgent
from .fine_tuning_agent import FineTuningAgent
from .modeling_agent import ModelingAgent
from .researcher_agent import ResearcherAgent
from .review_agent import ReviewAgent


_SYSTEM_PROMPT = """\
# Role & Objective
You are the AIML Scientist — the lead orchestrator of an autonomous AutoML
discovery platform. You direct a team of specialist sub-agents to produce
the highest-quality predictive model for the user's stated problem, following
the CRISP-DM lifecycle with rigorous planning and iterative improvement.

# Your Team
  • EDA Agent        — profiles data, creates charts, runs statistical analyses.
  • FE Agent         — builds and transforms datasets (50+ operations).
  • Modeling Agent   — trains AutoML baselines, tunes, ensembles, explains.
  • Review Agent     — critiques runs, flags leakage and overfitting.
  • Fine Tuning Agent— acts on critique to lift scores iteratively.
  • Researcher Agent — searches the web for domain knowledge and benchmarks.
  • Drift Agent      — detects distributional shifts between datasets.

════════════════════════════════════════════════════════════════════════
# PLANNING MANDATE — DO THIS FIRST, BEFORE ANY DELEGATION

Before dispatching ANY sub-agent, call record_observation with a
structured DISCOVERY PLAN. Use this exact template:

  ## DISCOVERY PLAN: [5-word problem description]

  **Task type:** Classification | Regression | Time-Series Forecasting
  **Primary metric:** [AUC-ROC | F1_weighted | R² | RMSE | MAE]
  **Success threshold:** [target metric value or business requirement]

  ### Problem Statement
  [One sentence: what we predict, for whom, at what granularity]

  ### Data Assessment
  - Datasets: [names, row counts, key columns]
  - Target column: [name]
  - Time column: [name if present, else "none"]
  - Key risks: [leakage candidates, data quality issues, imbalance]

  ### Feature Engineering Strategy
  [Priority operations in order with rationale]

  ### Modeling Strategy
  [Model families to prioritize and why; split approach]

  ### Evaluation Strategy
  [Holdout type; CV strategy; stopping criterion]

  ### Iteration Plan
  - Round 1: [specific improvement idea]
  - Round 2: [specific improvement idea]
  - Stop when: [plateau criterion — e.g., <1% gain over 2 consecutive rounds]

  ### Hypotheses to Test
  [2-3 testable hypotheses tied to specific feature or model choices]

UPDATE the plan (via a new record_observation) every time you rewind to
an earlier phase. Log what changed and why.

════════════════════════════════════════════════════════════════════════
# REASONING PROTOCOL — THINK BEFORE EVERY MAJOR DECISION

Before each delegation tool call, record one reasoning step:

  "Observation from last step: [what happened].
   Decision: I will [action] because [evidence].
   Expected outcome: [specific metric or artifact].
   If I instead see [failure signal] I will [fallback plan]."

This Thought → Action → Expected-Observation loop prevents silent
failures where the pipeline advances despite a broken upstream step.
Write this as a record_observation BEFORE the delegation call.

════════════════════════════════════════════════════════════════════════
# AIML LIFECYCLE (modified CRISP-DM)

Call set_phase() at the start of every phase and every rewind.
Phase enforcement is SOFT — revisit any phase when evidence demands it.

  ## 1. business_understanding
  Entry:  User provides dataset(s) and a question.
  Goal:   Confirm target column, task type, metric, and business constraints.
          Write the Discovery Plan here as record_observation.
  Ask_user ONLY when: 2+ plausible targets exist with different business
          meaning, OR domain-specific definitions only the user can give.
          Always include your recommended answer and 1-2 alternatives.
  Exit:   One-sentence problem statement is written. Discovery Plan exists.

  ## 2. data_understanding
  Entry:  Discovery Plan exists. Target and task type are confirmed.
  Goal:   Understand every column — distributions, missingness, cardinality,
          correlation with target, temporal structure.
          Delegation brief MUST include: dataset IDs, target column, task
          type, specific questions to answer (leakage check? seasonality?).
  Delegate: EDA Agent. Optionally Researcher Agent for domain gaps.
  Exit:   You know which features carry signal, which are leakage risks,
          and what transformations the FE Agent should apply.

  ## 3. data_preparation
  Entry:  EDA findings and Feature Engineering Strategy are in the notebook.
  Goal:   Produce one or more modelling-ready datasets.
          For forecasting: aggregate → fill panel → lag/rolling features →
          lead targets. For classification with imbalance: SMOTE after encoding.
  Delegation brief MUST include: source dataset ID, target column,
          task type, ordered list of operations with params, expected output name.
  Delegate: Feature Engineering Agent.
  Exit:   Dataset has correct shape, target column exists, time/group columns
          present. Record the new dataset_id in the plan.

  ## 4. modeling
  Entry:  Modelling-ready dataset exists.
  Goal:   Train baseline → Review → Fine Tune at least twice.
  Delegation brief MUST include: dataset ID, target column, task type,
          time_column if forecasting, current best metric to beat.
  Delegate: Modeling Agent (baseline) → Fine Tuning Agent (≥2 rounds).
  Exit:   Best metric improved < 1 % over prior best for TWO consecutive
          rounds, OR Review explicitly flags data ceiling.

  ## 5. evaluation
  Entry:  ≥ 2 trained model runs exist.
  Goal:   Identify best run, explain why, flag remaining risks.
  Delegation brief MUST include: run IDs to compare, primary metric,
          specific questions (leakage? overfitting? robustness?).
  Delegate: Review Agent.
  Exit:   Best run_id named with concrete metrics and risk assessment.

  ## 6. iteration
  Entry:  Evaluation is complete.
  Goal:   Decide: loop back or finalize.
  Options:
    (a) Rewind to data_preparation — new features from Review critique.
    (b) Rewind to modeling — different algorithm or target framing.
    (c) Call finalize_strategy — publish the final report.
  When rewinding: call set_phase() first, update Discovery Plan, explain why.

════════════════════════════════════════════════════════════════════════
# OPERATING PRINCIPLES

## A. DELEGATE WITH PRECISION — 4 REQUIRED ELEMENTS
Every sub-agent dispatch must include all four:
  1. OBJECTIVE: one sentence on what the agent should produce.
  2. CONTEXT: dataset IDs, column names, task type, current best metric.
  3. EXPECTED OUTPUT: the specific artifact (dataset_id, run_id, findings).
  4. CONSTRAINTS: what the agent must NOT do (e.g. "do not train yet").
Vague instructions produce vague results. Be specific about every column.

## B. CHASE THE BEST METRICS — NEVER ACCEPT THE FIRST MODEL
After the baseline, call Review → Fine Tuning AT LEAST TWICE.
Stop only when BOTH conditions hold:
  (a) ≥ 2 Review + Fine Tuning rounds have completed, AND
  (b) The last round improved primary metric by < 1 % over prior best.
  OR Review explicitly flags data-ceiling quality.
For classification: target AUC-ROC and F1_weighted (both, not just accuracy).
For regression: target R² (higher) and RMSE (lower).

## C. KEEP A RUNNING NARRATIVE
Use record_observation liberally. Capture: hypotheses, dead ends,
surprises, metric comparisons. These become the final report.

## D. HANDLE BLOCKED FEATURE ENGINEERING OPERATIONS
If FE returns created_dataset_ids=[] or reports "tool interface limitation":
  DO NOT send the problem to Modeling — it cannot aggregate or transform.
  Re-dispatch FE with an explicit JSON example:
    {"source_dataset_id":"<id>","new_name":"monthly_demand",
     "operation":"groupby_aggregate",
     "params":{"group_by":["request_month","part_no"],
               "aggregations":{"qty":"sum","order_no":"nunique"}}}
  All per-operation args MUST be nested under "params".

## E. HANDLE MISSING TARGET COLUMNS
If Modeling Agent reports target column not found:
  set_phase("data_preparation") → re-dispatch FE with explicit
  groupby_aggregate (specify group_by, aggregations, new_name) →
  return to modeling with the new dataset_id.

## F. USE THE RESEARCHER FOR DOMAIN GAPS AND BENCHMARKS
Call delegate_to_researcher when you encounter:
  – An unfamiliar domain or product category
  – Uncertainty about whether a metric level is reasonable
  – A surprising result that needs external verification
  – Need to know state-of-the-art for this problem type
  Pass a specific, focused question — not a vague topic.

## G. DRIFT DETECTION FOR MONITORING OR SUSPICIOUS SHIFTS
Call delegate_to_drift_detection when:
  – The user has a reference (training) dataset AND a new (production) dataset
  – A model's live performance has degraded unexpectedly
  – You suspect the new data distribution differs from what was trained on

════════════════════════════════════════════════════════════════════════
# FINISH WITH finalize_strategy

The final report MUST be a structured markdown document covering:
  1. Problem statement (1 paragraph)
  2. Data summary: datasets, row counts, transformations applied
  3. Experiment log: run_id | primary metric | key features/changes
  4. Best model: run_id, metric value, explanation of why it won
  5. What worked (tied to specific evidence)
  6. What did NOT work and why
  7. Remaining risks: leakage suspicions, fragile features, overfitting
  8. Recommended next steps: deployment, monitoring, data collection
"""


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "set_phase",
                "description": (
                    "Move the run into a new AIML lifecycle phase. Soft guidance "
                    "only — call this BEFORE delegating so every step from the "
                    "next sub-agent is tagged with the right phase in the "
                    "exported notebook. You may rewind to an earlier phase "
                    "(e.g. back to data_preparation from modeling) when "
                    "Review or User feedback requires it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phase": {
                            "type": "string",
                            "enum": PHASE_IDS,
                            "description": "The lifecycle phase to enter.",
                        },
                        "rationale": {
                            "type": "string",
                            "description": (
                                "One sentence on why you are entering (or "
                                "rewinding to) this phase. Stored in the "
                                "notebook for the exported workbook."
                            ),
                        },
                    },
                    "required": ["phase"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": (
                    "Ask the user 1–4 clarifying questions. ONLY call this when the answer "
                    "materially changes your plan and you cannot infer it from the data. "
                    "Each question MUST include your recommended answer and at least one "
                    "alternative so the user can simply accept your suggestion."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "recommendation": {
                                        "type": "string",
                                        "description": "Your suggested answer if the user just accepts your judgement.",
                                    },
                                    "alternatives": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "1-3 alternative answers the user could pick instead.",
                                    },
                                    "explanation": {
                                        "type": "string",
                                        "description": "Plain-language context for why this choice matters.",
                                    },
                                },
                                "required": ["question", "recommendation"],
                            },
                        }
                    },
                    "required": ["questions"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delegate_to_eda",
                "description": "Dispatch the EDA Agent with focused instructions.",
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
                "name": "delegate_to_feature_engineering",
                "description": "Dispatch the Feature Engineering Agent with focused instructions.",
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
                "name": "delegate_to_modeling",
                "description": "Dispatch the Modeling Agent with focused instructions (dataset_ids, target_columns).",
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
                "name": "delegate_to_review",
                "description": "Dispatch the Review Agent to critique training runs.",
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
                "name": "delegate_to_fine_tuning",
                "description": "Dispatch the Fine Tuning Agent to iterate on improvements.",
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
                "name": "delegate_to_researcher",
                "description": (
                    "Dispatch the Researcher Agent to search the web for domain knowledge, "
                    "ML technique benchmarks, or to resolve data uncertainties. "
                    "Pass a specific, focused research question."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The specific research question to investigate.",
                        }
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_observation",
                "description": "Write your own observation/hypothesis to the shared notebook.",
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
                "name": "finalize_strategy",
                "description": "Submit the final comprehensive markdown report and end the run.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Full markdown strategy report.",
                        }
                    },
                    "required": ["summary"],
                },
            },
        },
    ]


class AimlScientist(BaseAgent):
    """Top-level orchestrator. Drives the full multi-agent run as a generator."""

    name = "scientist"
    display_name = "AIML Scientist"

    def __init__(self, client, deployment: str, context: AgentContext) -> None:
        super().__init__(client, deployment, context)
        self.strategy_summary: str = ""
        self._messages: list[dict] = []

    def run(self) -> Generator[AutopilotStep, list[str] | None, None]:
        project = self._ctx.store.get_project(self._ctx.project_id)
        datasets = self._ctx.list_datasets()
        dataset_index = "\n".join(
            f"- id={d.id} name={d.name} rows={d.row_count} cols={d.column_count} type={d.source_type}"
            for d in datasets
        ) or "(no datasets registered)"

        user_prompt = (
            f"Project: **{project.name}**\n"
            f"Datasets ({len(datasets)}):\n{dataset_index}\n\n"
            f"User-stated goal: {self._ctx.user_goal or '(none provided)'}\n\n"
            "Decide your plan and begin. Remember: only ask the user when "
            "absolutely necessary and always include your own recommendation."
        )

        self._messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        self._persist_message(self._messages[0])
        self._persist_message(self._messages[1])
        yield from self._run_loop_iterations()

    # ------------------------------------------------------------------
    # Resume / continue support
    # ------------------------------------------------------------------

    def load_messages(self, messages: list[dict], strategy_summary: str = "") -> None:
        """Rehydrate a prior conversation so continue_with() can pick up."""
        self._messages = list(messages)
        self.strategy_summary = strategy_summary

    def continue_with(
        self, user_message: str
    ) -> Generator[AutopilotStep, list[str] | None, None]:
        """Append a follow-up user message and resume the orchestrator loop."""
        if not self._messages:
            # If we are continuing after a fresh load with no system prompt,
            # seed one so the LLM still has its role.
            self._messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
            self._persist_message(self._messages[0])
        followup = {"role": "user", "content": user_message}
        self._messages.append(followup)
        self._persist_message(followup)
        yield from self._run_loop_iterations()

    # ------------------------------------------------------------------
    # Loop driver — handles ask_user pauses and sub-agent forwarding.
    # ------------------------------------------------------------------

    def _run_loop_iterations(
        self,
    ) -> Generator[AutopilotStep, list[str] | None, None]:
        messages = self._messages
        tools = _tools()

        for iteration in range(60):
            if self._ctx.should_stop:
                log.info("Scientist loop stopping | iteration=%d (stop requested)", iteration)
                break
            log.debug(
                "Scientist LLM call | iteration=%d messages=%d",
                iteration, len(messages),
            )
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            choice = response.choices[0]
            log.debug(
                "Scientist LLM response | finish_reason=%s has_tool_calls=%s",
                choice.finish_reason, bool(choice.message.tool_calls),
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
                yield self._step("thought", "AIML Scientist — Reasoning", choice.message.content)

            if choice.finish_reason == "stop":
                log.info("Scientist loop finished | iterations=%d", iteration + 1)
                break
            if choice.finish_reason != "tool_calls":
                log.warning("Scientist unexpected finish_reason=%r", choice.finish_reason)
                continue

            terminate = False
            for tc in choice.message.tool_calls:
                name = tc.function.name
                args: dict[str, Any] = json.loads(tc.function.arguments or "{}")

                log.info("Scientist tool_call | name=%s args_preview=%s", name, str(args)[:300])
                yield self._step(
                    "tool_call",
                    f"[Scientist] {name}",
                    json.dumps(args, indent=2),
                )

                tool_content, terminate_flag = yield from self._dispatch(name, args)

                if tool_content is not None:
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    }
                    messages.append(tool_msg)
                    self._persist_message(tool_msg)
                if terminate_flag:
                    terminate = True
                    break

            if terminate:
                break

    # ------------------------------------------------------------------
    # Tool dispatch — note this is a generator because some tools
    # ask the user (which suspends) or yield from sub-agents.
    # ------------------------------------------------------------------

    def _dispatch(
        self, name: str, args: dict
    ) -> Generator[AutopilotStep, list[str] | None, tuple[str | None, bool]]:
        if name == "set_phase":
            target = (args.get("phase") or "").strip()
            rationale = (args.get("rationale") or "").strip()
            previous = self._ctx.current_phase
            resolved = self._ctx.set_phase(target)
            phase_meta = PHASE_BY_ID.get(resolved, {})
            log.info(
                "Scientist phase transition | %s → %s rationale=%r",
                previous, resolved, rationale[:120],
            )
            # Tag the transition step with the NEW phase so it appears under
            # the destination section in the exported notebook.
            transition_step = self._step(
                "phase_transition",
                f"Phase → {phase_meta.get('title', resolved)}",
                rationale or phase_meta.get("description", ""),
                data={
                    "from_phase": previous,
                    "to_phase": resolved,
                    "rationale": rationale,
                },
            )
            yield transition_step
            if rationale:
                self._ctx.notebook.append(
                    f"[Phase: {phase_meta.get('title', resolved)}] {rationale}"
                )
            return (
                json.dumps({
                    "previous_phase": previous,
                    "current_phase": resolved,
                    "title": phase_meta.get("title", resolved),
                }),
                False,
            )

        if name == "ask_user":
            content = yield from self._ask_user(args)
            return content, False

        if name == "delegate_to_eda":
            log.info("Scientist delegating → EDA Agent")
            sub = EdaAgent(self._client, self._deployment, self._ctx)
            summary = yield from sub.run(args.get("instructions", ""))
            log.info("EDA Agent returned | summary_keys=%s", list(summary.keys()) if isinstance(summary, dict) else type(summary))
            return json.dumps(to_json_safe(summary)), False

        if name == "delegate_to_feature_engineering":
            log.info("Scientist delegating → Feature Engineering Agent")
            sub = FeatureEngineeringAgent(self._client, self._deployment, self._ctx)
            summary = yield from sub.run(args.get("instructions", ""))
            log.info("FE Agent returned | summary_keys=%s", list(summary.keys()) if isinstance(summary, dict) else type(summary))
            return json.dumps(to_json_safe(summary)), False

        if name == "delegate_to_modeling":
            log.info("Scientist delegating → Modeling Agent")
            sub = ModelingAgent(self._client, self._deployment, self._ctx)
            summary = yield from sub.run(args.get("instructions", ""))
            log.info("Modeling Agent returned | summary_keys=%s", list(summary.keys()) if isinstance(summary, dict) else type(summary))
            return json.dumps(to_json_safe(summary)), False

        if name == "delegate_to_review":
            log.info("Scientist delegating → Review Agent")
            sub = ReviewAgent(self._client, self._deployment, self._ctx)
            summary = yield from sub.run(args.get("instructions", ""))
            log.info("Review Agent returned | summary_keys=%s", list(summary.keys()) if isinstance(summary, dict) else type(summary))
            return json.dumps(to_json_safe(summary)), False

        if name == "delegate_to_fine_tuning":
            log.info("Scientist delegating → Fine Tuning Agent")
            sub = FineTuningAgent(self._client, self._deployment, self._ctx)
            summary = yield from sub.run(args.get("instructions", ""))
            log.info("Fine Tuning Agent returned | summary_keys=%s", list(summary.keys()) if isinstance(summary, dict) else type(summary))
            return json.dumps(to_json_safe(summary)), False

        if name == "delegate_to_researcher":
            log.info("Scientist delegating → Researcher Agent")
            sub = ResearcherAgent(self._client, self._deployment, self._ctx)
            summary = yield from sub.run(args.get("question", ""))
            log.info("Researcher Agent returned | summary_keys=%s", list(summary.keys()) if isinstance(summary, dict) else type(summary))
            return json.dumps(to_json_safe(summary)), False

        if name == "record_observation":
            text = (args.get("text") or "").strip()
            if text:
                self._ctx.notebook.append(f"[Scientist] {text}")
                yield self._step("observation", "Scientist observation", text)
            return json.dumps({"recorded": True}), False

        if name == "finalize_strategy":
            summary = args.get("summary", "")
            self.strategy_summary = summary
            session = getattr(self._ctx, "session", None)
            if session is not None:
                try:
                    session.set_strategy_summary(summary)
                except Exception as exc:  # pragma: no cover — defensive
                    log.warning("Could not persist strategy_summary: %s", exc)
            yield self._step("summary", "Final Strategy Report", summary)
            return json.dumps({"status": "done"}), True

        return json.dumps({"error": f"Unknown tool: {name}"}), False

    # ------------------------------------------------------------------
    # ask_user — pause and resume via send()
    # ------------------------------------------------------------------

    def _ask_user(
        self, args: dict
    ) -> Generator[AutopilotStep, list[str] | None, str]:
        raw_questions = args.get("questions") or []
        normalised: list[dict[str, Any]] = []
        for q in raw_questions:
            if isinstance(q, str):
                normalised.append(
                    {"question": q, "recommendation": "", "alternatives": [], "explanation": ""}
                )
            elif isinstance(q, dict):
                normalised.append(
                    {
                        "question": q.get("question", ""),
                        "recommendation": q.get("recommendation", ""),
                        "alternatives": q.get("alternatives", []) or [],
                        "explanation": q.get("explanation", ""),
                    }
                )

        ask_step = self._step(
            "ask",
            "AIML Scientist has questions for you",
            f"{len(normalised)} question(s) — each with a recommended answer",
            data={"questions": normalised},
        )
        answers: list[str] | None = yield ask_step
        # If the user accepted defaults via empty answers, use the scientist's
        # own recommendations so the run continues coherently.
        if not answers:
            answers = [q.get("recommendation", "") for q in normalised]
        else:
            answers = [
                a if a.strip() else normalised[i].get("recommendation", "")
                for i, a in enumerate(answers)
            ]

        session = getattr(self._ctx, "session", None)
        if session is not None:
            try:
                session.patch_ask_answers(ask_step.index, list(answers))
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("Could not persist ask answers: %s", exc)

        formatted = "\n".join(
            f"Q{i+1}: {q['question']}\n"
            f"Scientist's recommendation: {q.get('recommendation', '')}\n"
            f"User's answer: {answers[i] if i < len(answers) else ''}"
            for i, q in enumerate(normalised)
        )
        for i, q in enumerate(normalised):
            self._ctx.user_answers.append(
                {
                    "question": q["question"],
                    "answer": answers[i] if i < len(answers) else "",
                    "scientist_recommended": q.get("recommendation", ""),
                }
            )
        return json.dumps({"user_answers": formatted})
