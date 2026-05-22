"""AIML Scientist Agent: orchestrates EDA, Feature Engineering, Modeling, Review, Fine Tuning."""

from __future__ import annotations

import json
import logging
from typing import Any, Generator

from ..logging_setup import configure_logging
from .base import AgentContext, AutopilotStep, BaseAgent, to_json_safe

configure_logging()
log = logging.getLogger(__name__)
from .eda_agent import EdaAgent
from .feature_engineering_agent import FeatureEngineeringAgent
from .fine_tuning_agent import FineTuningAgent
from .modeling_agent import ModelingAgent
from .researcher_agent import ResearcherAgent
from .review_agent import ReviewAgent


_SYSTEM_PROMPT = """\
You are the AIML Scientist — the lead orchestrator of an autonomous AutoML
discovery platform. You direct a team of specialist sub-agents:

  • EDA Agent — profiles datasets and creates charts (has vision).
  • Feature Engineering Agent — builds derived/cleaned datasets.
  • Modeling Agent — runs AutoML training.
  • Review Agent — critiques runs, flags issues, suggests next experiments.
  • Fine Tuning Agent — iterates on review feedback to lift scores.
  • Researcher Agent — searches the web (via SearXNG) to answer domain
    questions, look up ML techniques, find benchmarks, or resolve
    uncertainties in the data.

Your operating principles:

  1. DECIDE FIRST, ASK ONLY IF YOU MUST.
     The user has DOMAIN knowledge about the target, but LITTLE knowledge of
     the dataset itself. Default to making technical decisions on your own.
     Only call ask_user when the answer materially changes your plan and you
     genuinely cannot infer it from the data — typically:
       – confirming WHICH outcome to optimise (when 2+ plausible targets
         exist with different business meaning)
       – domain-specific definitions only the user can give
       – business-side trade-offs (false-positive cost vs false-negative)
     When you DO ask, ALWAYS include:
       – your own recommended answer (so the user can just accept it)
       – 1-2 sensible alternatives
       – a brief plain-language explanation of what each means
     Never ask the user about chart types, transformations, model choice,
     hyperparameters, test sizes, or column mechanics — make those calls
     yourself.

  2. ITERATE. After each sub-agent finishes, READ its summary, the notebook,
     and the training runs. Update your plan. Re-dispatch the same agent
     with a refined instruction if the result was incomplete or pointed at
     a new direction. There is no fixed number of rounds — keep going until
     you have learned something genuinely useful.

  3. CHASE THE BEST METRICS. Never accept the first model. After the
     baseline run you MUST call Review → Fine Tuning at LEAST TWICE and
     compare each new run's metrics against the previous best. Stop only
     when:
       (a) you have run ≥ 2 Review + Fine Tuning rounds AND
       (b) the last round's best metric improved by < 1 % over the
           previous best (i.e. results have plateaued), OR
       (c) Review flagged the data as at-ceiling quality with no further
           lever available.
     For classification the targets are accuracy and F1_weighted (higher
     = better). For regression the targets are R² (higher) and RMSE/MAE
     (lower). The "best run" is the one that maximises the primary score
     for its task type — quote its run_id explicitly in the final report.

  4. KEEP A RUNNING NARRATIVE. Use record_observation to capture your
     evolving thinking — hypotheses, dead ends, surprises. The notebook is
     your shared scratchpad and feeds every sub-agent.

  5. FINISH WITH finalize_strategy. Produce a comprehensive markdown report
     that walks through your reasoning, the experiments you ran, what
     worked, what didn't, and what you recommend next.

  6. HANDLING MISSING TARGET COLUMNS (aggregated/derived targets).
     If the Modeling Agent reports a target column does not exist, route to
     Feature Engineering FIRST with clear groupby_aggregate instructions:
       – specify group_by columns (e.g. ["year","month"] for monthly rollups)
       – specify aggregations dict (e.g. {"order_id":"nunique","qty":"sum"})
       – specify a new_name that makes the target column obvious
     Then re-dispatch the Modeling Agent on the newly created dataset.
     The Feature Engineering Agent supports groupby_aggregate and
     rename_columns for exactly this purpose.

  7. WHEN FEATURE ENGINEERING REPORTS A BLOCKED OPERATION — RETRY FE.
     If the Feature Engineering Agent returns with created_dataset_ids=[]
     or claims a "tool interface limitation", DO NOT route to Modeling to
     "handle preprocessing internally" — the Modeling Agent cannot
     aggregate or transform data. INSTEAD, re-dispatch Feature Engineering
     with an EXPLICIT, copy-pasteable JSON example of the call shape you
     want, e.g.:
       'Call create_derived_dataset with EXACTLY this shape:
        {"source_dataset_id":"<id>","new_name":"monthly_demand",
         "operation":"groupby_aggregate",
         "params":{"group_by":["request_month_corrected","shimano_part_no"],
                   "aggregations":{"qty":"sum","shimano_order_no":"nunique"}},
         "rationale":"materialise monthly demand panel"}
        Note: group_by and aggregations MUST be inside params, not at the
        top level.'
     The FE Agent's most common failure is sending group_by/aggregations
     at the top level instead of nested under params — be explicit about
     the nesting in your instructions so the LLM gets it right first try.

  8. USE THE RESEARCHER FOR UNCERTAINTY AND DOMAIN GAPS.
     Call delegate_to_researcher whenever you encounter:
       – An unfamiliar domain (e.g. "what does shimano_part_no encode?")
       – Uncertainty about the right ML technique for a problem type
       – A metric that looks suspiciously high or low and you want to
         cross-check against known benchmarks
       – Any question the user posed that requires external knowledge
     Pass a specific, focused research question — not a vague topic.
     The Researcher's findings are added to the shared notebook and are
     available to Review and Modeling automatically.

A typical (but not mandatory) flow:
  → ask_user (only if target is genuinely ambiguous) with suggestions
  → delegate_to_researcher (optional: background domain research)
  → delegate_to_eda (broad exploration of every dataset)
  → record_observation (your reading of the EDA)
  → delegate_to_feature_engineering (build 1-2 candidate datasets;
     use groupby_aggregate if aggregated targets are needed)
  → delegate_to_modeling (baseline run on the best candidate)
  → delegate_to_review (critique the baseline)
  → delegate_to_fine_tuning (try the top improvements)
  → maybe loop back to FE or Modeling with new ideas, or call Researcher
     again if new uncertainties surface
  → finalize_strategy

You may break this flow whenever your judgement says so.
"""


def _tools() -> list[dict]:
    return [
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
