"""Modeling Agent: trains AutoML runs and visualises their performance."""

from __future__ import annotations

import json
import logging
from typing import Any, Generator

from ..diagnostics import (
    build_diagnostic_figures,
    build_feature_importance_figure,
    build_leaderboard_figure,
    build_primary_diagnostic_figure,
    build_residuals_over_time_figure,
    build_run_comparison_figure,
)
from ..ingestion import load_dataset
from ..logging_setup import configure_logging
from ..profiling import profile_dataframe
from ..reporting import build_markdown_report
from ..training import TrainingSettings, train_automl_stream
from .base import AgentContext, AutopilotStep, BaseAgent, to_json_safe, vision_tool_content

configure_logging()
log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
# Role & Objective
You are the Modeling Agent — the team's hands-on ML engineer with deep
expertise in model selection, validation strategy, diagnostics, and
iterative improvement. You have final authority over training decisions.

════════════════════════════════════════════════════════════════════════
# REASONING PROTOCOL — THINK BEFORE EVERY MAJOR DECISION

Before each train_model call, write a reasoning observation:
  "Dataset has [N rows, M features]. Task type: [type].
   Best model so far: [name / metric or 'none'].
   I will train [model subset] because [evidence from profile/EDA].
   I expect [metric range]. Key risk: [leakage? imbalance? small data?]."

After every chart, write: "I see [description]. This means [implication]."

════════════════════════════════════════════════════════════════════════
# STEP 1 — ALWAYS START WITH inspect_dataset

Confirm:
  • Target column exists and has the expected dtype.
  • Time column present for forecasting (look for: date, month, ds,
    order_date, request_month, period, week, timestamp).
  • Lead/lag/rolling features FE created (cols ending in _lag_, _roll_,
    _lead_) — if missing and task is forecasting, STOP and report.
  • Class distribution for classification — severe imbalance (>5:1) changes
    your metric choice (use F1/AUC not accuracy) and training strategy.

════════════════════════════════════════════════════════════════════════
# STEP 2 — CHOOSE THE RIGHT TRAINING MODE

## For FORECASTING / TIME-SERIES
  ALWAYS set time_column. This forces chronological holdout.
  NEVER use random split for temporal data — it inflates all metrics.
  Train on lead targets (qty_lead_1, qty_lead_3) that FE created.
  If no lead targets exist → call done() with rationale="BLOCKED: run
  create_lead_target + create_lag_features in FE first."
  Use test_size=0.2 for long series (>24 periods), 0.3 for short.

## For CLASSIFICATION
  Severe imbalance (>5:1): set class_weight="balanced" in train_model.
  Primary metric: AUC-ROC and F1_weighted (not accuracy).
  Use stratified hold-out (default when no time_column).

## For REGRESSION
  Primary metric: R² (higher) and RMSE (lower).
  Check for skewed target: if normality_test showed skewness, request
  target_log_transform from FE before training.
  Use random hold-out (default).

════════════════════════════════════════════════════════════════════════
# STEP 3 — MODEL SELECTION DECISION TREE

## Tabular data, any size
  First run: train ALL models (omit include_models) OR use a smart subset:
    Fast first pass (include_models):
      ["Baseline (Mean)", "Ridge", "Random Forest", "Hist Gradient Boosting",
       "XGBoost", "LightGBM"]   ← covers linear + tree + boosting families
    If first pass shows tree models dominating: add CatBoost, Extra Trees.
    If linear model is competitive: add ElasticNet, Bayesian Ridge, Lasso.
    Skip slow models (SVR RBF, SVC RBF, MLP) when N > 50k rows.

## Small dataset (< 2k rows)
  Risk: overfitting. Prefer cross_validate_model before full train.
  Prefer: Ridge, Lasso, ElasticNet, Random Forest with max_depth limit.
  Avoid: deep trees, KNN with small k, MLP.

## Time-series / forecasting
  Preferred: Hist Gradient Boosting, XGBoost, LightGBM (handle lags well).
  Consider: train_arima for short univariate series (<500 obs).
  Avoid: KNN, SVC (no temporal structure awareness).

## Very high cardinality features
  Preferred: tree-based models (handle categories natively after encoding).
  After encoding: LightGBM, CatBoost (native categorical support).

════════════════════════════════════════════════════════════════════════
# STEP 4 — VISUALISE AND INTERPRET

After every train_model, review the primary diagnostic chart automatically
returned. Then call create_model_chart for:
  • leaderboard — always: shows all model rankings, spots leakage winner.
  • predicted_vs_actual — regression: should scatter around y=x line.
  • forecast — time-series: actual vs predicted lines should track closely.
  • residuals_over_time — look for time-correlated residuals (concept drift).
  • feature_importance — always for tree models: flag if one feature
    dominates (> 70 % = likely leakage proxy).
  • confusion_matrix — classification: check minority class recall.

## Diagnostic Interpretation
  | Pattern                              | Meaning                     | Action                          |
  |--------------------------------------|-----------------------------|---------------------------------|
  | Predictions flat at mean             | Model found no signal       | FE needed; check target framing |
  | Residuals trend over time            | Concept drift               | Add time features; rolling model|
  | One feature > 70 % importance        | Leakage proxy               | Drop feature; retrain           |
  | Train metric >> CV metric (>10% gap) | Overfitting                 | Regularise; reduce features     |
  | All models tied at baseline          | Target has no predictors    | Recheck target; different grain |
  | High recall/precision but low AUC   | Threshold artefact          | Check class balance; calibrate  |

Use compare_runs to put multiple experiments side-by-side.

════════════════════════════════════════════════════════════════════════
# STEP 5 — CROSS-VALIDATION AND HYPERPARAMETER TUNING

## When to use cross_validate_model
  • Dataset < 5k rows: CV gives more reliable estimates than one split.
  • Comparing two models head-to-head on the same folds.
  • Checking fold-to-fold variance (std > 0.05 = unstable).
  Set time_column for TimeSeriesSplit; omit for KFold.
  Set class_weight="balanced" for imbalanced classification.

## When to tune hyperparameters
  After identifying the best model FAMILY from the leaderboard.
  Use tune_hyperparameters (RandomizedSearchCV) for a quick search.
  Use tune_hyperparameters_optuna (Bayesian HPO) for deeper search.
  Lock in best params via custom_models in a follow-up train_model.
  Do NOT tune slow models (SVR, MLP) on large datasets.

## When to build ensembles
  After ≥ 3 diverse models are trained.
  Voting ensemble: best when models are similarly strong but use different
    representations (e.g., tree + linear + boosting).
  Stacking ensemble: best when one model can learn from others' errors.
  Typically adds +1-3 % over the best single model.

════════════════════════════════════════════════════════════════════════
# RESEARCHER INTEGRATION

Call spawn_researcher when you need external knowledge to make better decisions:
  • At the START of a run — look up expected metric ranges or state-of-the-art
    approaches for [task_type] in this domain (e.g., "best models for demand
    forecasting tabular data", "typical AUC for churn prediction").
  • When a metric seems suspiciously high or unexpectedly low — verify if
    that level is known to be achievable for this problem type.
  • When you are unsure which model family is best for an unusual feature set.
  • When a diagnostic pattern is ambiguous and you want literature support.
Pass a specific, focused question — not a vague topic.

════════════════════════════════════════════════════════════════════════
# STEP 6 — EXPLAINABILITY

Call explain_model (SHAP) when:
  • The Scientist or user wants to understand feature contributions.
  • Feature importance shows a suspicious dominant feature.
  • The task is classification and you want to explain a specific prediction.
SHAP TreeExplainer works for all tree models (fast).
SHAP LinearExplainer for Ridge/Lasso/Logistic.
SHAP KernelExplainer for others (slow on large datasets).

════════════════════════════════════════════════════════════════════════
# STOP CRITERIA

Call done(summary) when EITHER:
  (a) You have trained a strong baseline + at least one improvement run.
  (b) You are BLOCKED (no valid dataset, missing lead targets, etc.).

In done(summary) always include:
  strongest_run_id, primary_metric, metric_name, concerns (array),
  rationale (citing specific chart observations and metric values).

Available models (pass names verbatim in include_models):
  Classification: Baseline (Majority), Logistic Regression, SGD Classifier,
    Linear SVC, Gaussian Naive Bayes, Bernoulli Naive Bayes, LDA, QDA,
    Decision Tree, Extra Trees, Random Forest, AdaBoost, Gradient Boosting,
    Hist Gradient Boosting, K-Nearest Neighbors, SVC (RBF), MLP, Bagging,
    XGBoost*, LightGBM*, CatBoost*
  Regression: Baseline (Mean), Linear Regression, Ridge, Lasso, ElasticNet,
    Bayesian Ridge, Huber Regressor, SGD Regressor, Decision Tree,
    Extra Trees, Random Forest, AdaBoost, Gradient Boosting,
    Hist Gradient Boosting, K-Nearest Neighbors, Linear SVR, SVR (RBF),
    MLP, Bagging, XGBoost*, LightGBM*, CatBoost*
  (* installed automatically if the library is present)
"""


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "inspect_dataset",
                "description": (
                    "Return all column names in a dataset. Call this FIRST "
                    "to confirm the target exists and find the time column."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"dataset_id": {"type": "string"}},
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_run",
                "description": (
                    "Return metadata for a saved run (target, task type, best metrics, "
                    "leaderboard, whether time-series). Use to look up details before "
                    "calling create_model_chart or compare_runs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "required": ["run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "train_model",
                "description": (
                    "Run AutoML training. Set time_column to the date column "
                    "to force a chronological train/test split (the proper "
                    "backtest for forecasting). Omit time_column for "
                    "non-temporal problems (random split)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "target_column": {"type": "string"},
                        "test_size": {"type": "number", "description": "Default 0.2"},
                        "random_state": {"type": "integer", "description": "Default 42"},
                        "time_column": {
                            "type": "string",
                            "description": (
                                "Optional. If set, the split is CHRONOLOGICAL — "
                                "rows sorted by this column, last test_size "
                                "fraction held out. Required for honest "
                                "forecasting backtest."
                            ),
                        },
                        "include_models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional subset of standard model names to run. "
                                "If omitted, ALL available models run. Use to skip "
                                "slow models on large datasets, e.g. "
                                "[\"Hist Gradient Boosting\", \"Random Forest\", \"LightGBM\"]."
                            ),
                        },
                        "custom_models": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "class": {
                                        "type": "string",
                                        "description": "Dotted import path, e.g. 'xgboost.XGBRegressor'.",
                                    },
                                    "params": {"type": "object"},
                                },
                                "required": ["class"],
                            },
                            "description": (
                                "Additional sklearn-compatible models to include. "
                                "These run alongside (or instead of) standard models."
                            ),
                        },
                        "class_weight": {
                            "type": "string",
                            "enum": ["balanced", "none"],
                            "description": (
                                "Set 'balanced' for imbalanced classification. "
                                "Applies class_weight to all classifiers that support it "
                                "(LR, RF, ET, Decision Tree, SGD, SVC, etc.). "
                                "Use instead of or combined with SMOTE resampling."
                            ),
                        },
                    },
                    "required": ["dataset_id", "target_column"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_model_chart",
                "description": (
                    "Render a diagnostic chart for a saved run and receive "
                    "the image back for visual analysis. You have vision — "
                    "describe what you see."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "chart_type": {
                            "type": "string",
                            "enum": [
                                "predicted_vs_actual",
                                "forecast",
                                "residuals",
                                "residuals_over_time",
                                "confusion_matrix",
                                "feature_importance",
                                "leaderboard",
                            ],
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "For feature_importance: how many to show (default 20).",
                        },
                    },
                    "required": ["run_id", "chart_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_runs",
                "description": (
                    "Render a bar chart comparing the primary metric across "
                    "multiple saved run_ids."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "metric": {
                            "type": "string",
                            "description": "Metric key (e.g. r2, rmse, f1_weighted). Auto-picks if omitted.",
                        },
                    },
                    "required": ["run_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_finding",
                "description": "Write a short note to the shared notebook.",
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
                    "search the web via SearXNG. Use this to look up benchmarks, "
                    "technique guidance, or domain context relevant to the modeling task."
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
                "name": "build_ensemble",
                "description": (
                    "Build a VotingClassifier or VotingRegressor (soft-voting for classifiers) "
                    "OR a StackingClassifier / StackingRegressor from the best models in completed runs. "
                    "Use this as a final step when individual models have plateaued to squeeze out "
                    "extra performance through combination. "
                    "ensemble_type: 'voting' (fast, averages predictions) or "
                    "'stacking' (slow, trains a meta-learner on top)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "target_column": {"type": "string"},
                        "model_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Names of the models to include in the ensemble. "
                                "Should be 2-5 diverse models that performed well individually."
                            ),
                        },
                        "ensemble_type": {
                            "type": "string",
                            "enum": ["voting", "stacking"],
                            "description": "Default 'voting'. Use 'stacking' for a meta-learner.",
                        },
                        "time_column": {
                            "type": "string",
                            "description": "If set, uses chronological split.",
                        },
                        "test_size": {"type": "number", "description": "Default 0.2"},
                        "class_weight": {
                            "type": "string",
                            "enum": ["balanced", "none"],
                        },
                    },
                    "required": ["dataset_id", "target_column", "model_names"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cross_validate_model",
                "description": (
                    "Run k-fold (or TimeSeriesSplit for forecasting) cross-validation on "
                    "a single model and return mean ± std of the primary metric. "
                    "Use this when you want a more robust performance estimate before "
                    "committing to a full train_model run, or when the dataset is small."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "target_column": {"type": "string"},
                        "model_name": {
                            "type": "string",
                            "description": "One of the standard model names (e.g. 'Random Forest', 'XGBoost').",
                        },
                        "n_splits": {
                            "type": "integer",
                            "description": "Number of CV folds (default 5).",
                        },
                        "time_column": {
                            "type": "string",
                            "description": "If set, uses TimeSeriesSplit (chronological CV).",
                        },
                        "class_weight": {
                            "type": "string",
                            "enum": ["balanced", "none"],
                            "description": "Set 'balanced' for class-imbalance. Applies to classifiers only.",
                        },
                    },
                    "required": ["dataset_id", "target_column", "model_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tune_hyperparameters",
                "description": (
                    "Run randomized hyperparameter search on a specific model over a given dataset. "
                    "Returns the best params and their cross-validated score. "
                    "Use after baseline training when you want to squeeze more out of the best model."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "target_column": {"type": "string"},
                        "model_name": {
                            "type": "string",
                            "description": "Name of the model to tune.",
                        },
                        "n_iter": {
                            "type": "integer",
                            "description": "Number of random search iterations (default 20).",
                        },
                        "n_splits": {
                            "type": "integer",
                            "description": "CV folds for the inner loop (default 3).",
                        },
                        "time_column": {
                            "type": "string",
                            "description": "If set, uses TimeSeriesSplit for CV.",
                        },
                        "param_grid": {
                            "type": "object",
                            "description": (
                                "Optional custom param distributions. "
                                "If omitted, a sensible default grid is used for the named model."
                            ),
                        },
                    },
                    "required": ["dataset_id", "target_column", "model_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "done",
                "description": "Finish modeling. Report the runs you executed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_ids": {"type": "array", "items": {"type": "string"}},
                        "strongest_run_id": {"type": "string"},
                        "rationale": {"type": "string"},
                        "concerns": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["rationale"],
                },
            },
        },
    ]


class ModelingAgent(BaseAgent):
    name = "modeling"
    display_name = "Modeling Agent"

    def __init__(self, client, deployment: str, context: AgentContext) -> None:
        super().__init__(client, deployment, context)
        self._summary: dict[str, Any] = {}
        self._run_ids: list[str] = []

    def run(
        self, instructions: str
    ) -> Generator[AutopilotStep, list[str] | None, dict[str, Any]]:
        log.info("Modeling Agent starting | instructions=%s", instructions[:200])
        yield self._step(
            "agent_start",
            "Modeling Agent dispatched",
            instructions or "(train the recommended candidates)",
        )

        datasets = self._ctx.list_datasets()
        dataset_index = "\n".join(
            f"- id={d.id} name={d.name} rows={d.row_count} cols={d.column_count}"
            for d in datasets
        ) or "(no datasets)"

        user_prompt = (
            f"Scientist's instructions:\n{instructions}\n\n"
            f"Available datasets:\n{dataset_index}\n\n"
            f"Notebook so far:\n{self._ctx.notebook_text()}\n\n"
            f"Existing training runs:\n{self._ctx.training_runs_summary()}\n\n"
            "Inspect first. If this looks like forecasting, use time_column "
            "in train_model. Visualise with create_model_chart."
        )

        yield from self._drive_loop(user_prompt)

        if self._run_ids:
            self._summary.setdefault("run_ids", self._run_ids)

        log.info(
            "Modeling Agent finished | run_ids=%s summary_keys=%s",
            self._run_ids, list(self._summary.keys()),
        )
        yield self._step("agent_end", "Modeling Agent finished", "")
        return self._summary or {"rationale": "Modeling agent ended without summary."}

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

        for _ in range(30):
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
                yield self._step("thought", "Modeling Agent — Reasoning", choice.message.content)

            if choice.finish_reason == "stop":
                break
            if choice.finish_reason != "tool_calls":
                continue

            terminate = False
            for tc in choice.message.tool_calls:
                name = tc.function.name
                args: dict[str, Any] = json.loads(tc.function.arguments or "{}")

                log.info("Modeling tool_call | name=%s", name)
                yield self._step(
                    "tool_call",
                    f"[Modeling Agent] {name}",
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
                else:
                    tool_content, extra_step, terminate_flag = self._dispatch(name, args, tc.id)
                    if extra_step is not None:
                        yield extra_step
                    if tool_content is not None:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_content,
                        })
                    if terminate_flag:
                        terminate = True
                        break

            if terminate:
                break

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self, name: str, args: dict, tool_call_id: str
    ) -> tuple[str | list | None, AutopilotStep | None, bool]:
        if name == "inspect_dataset":
            return self._inspect(args.get("dataset_id", ""))
        if name == "inspect_run":
            return self._inspect_run(args.get("run_id", ""))
        if name == "train_model":
            return self._train(args)
        if name == "create_model_chart":
            return self._create_model_chart(args)
        if name == "compare_runs":
            return self._compare_runs(args)
        if name == "record_finding":
            text = (args.get("text") or "").strip()
            if text:
                self._ctx.notebook.append(f"[Modeling] {text}")
            return (
                json.dumps({"recorded": True}),
                self._step("observation", "Modeling note", text),
                False,
            )
        if name == "build_ensemble":
            return self._build_ensemble(args)
        if name == "cross_validate_model":
            return self._cross_validate(args)
        if name == "tune_hyperparameters":
            return self._tune(args)
        if name == "done":
            self._summary = to_json_safe(args)
            return json.dumps({"status": "noted"}), None, True
        return json.dumps({"error": f"Unknown tool: {name}"}), None, False

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _inspect(self, dataset_id: str) -> tuple[str, AutopilotStep | None, bool]:
        ds = self._ctx.find_dataset(dataset_id)
        if ds is None:
            log.warning("inspect_dataset | dataset_id=%r not found", dataset_id)
            return json.dumps({"error": f"Dataset '{dataset_id}' not found."}), None, False

        loaded = load_dataset(ds.file_path, ds.table_name)
        cols = list(loaded.dataframe.columns)
        log.info(
            "inspect_dataset | dataset=%s rows=%d columns(%d)=%s",
            ds.name, len(loaded.dataframe), len(cols), cols[:20],
        )
        return (
            json.dumps({
                "dataset_id": ds.id,
                "name": ds.name,
                "rows": len(loaded.dataframe),
                "column_count": len(cols),
                "columns": cols,
            }),
            None,
            False,
        )

    def _inspect_run(self, run_id: str) -> tuple[str, AutopilotStep | None, bool]:
        run = self._find_run(run_id)
        if run is None:
            log.warning("inspect_run | run_id=%r not found", run_id)
            return json.dumps({"error": f"Run '{run_id}' not found."}), None, False
        diag = run.get("diagnostics") or {}
        payload = {
            "run_id": run.get("run_id"),
            "task_type": run.get("task_type"),
            "target_column": run.get("target_column"),
            "best_model": run.get("best_model_name"),
            "best_metrics": run.get("best_metrics", {}),
            "is_time_series": bool(diag.get("is_time_series")),
            "n_test_points": diag.get("n_points"),
            "settings": run.get("settings", {}),
            "leaderboard_models": [
                e.get("model") for e in run.get("leaderboard", []) if e.get("status") == "success"
            ],
        }
        log.info("inspect_run | run_id=%s payload_keys=%s", run_id, list(payload.keys()))
        return json.dumps(to_json_safe(payload)), None, False

    def _train(self, args: dict) -> tuple[str | list, AutopilotStep | None, bool]:
        ds = self._ctx.find_dataset(args.get("dataset_id", ""))
        if ds is None:
            log.warning("train_model | dataset_id=%r not found", args.get("dataset_id"))
            return (
                json.dumps({"error": f"Dataset '{args.get('dataset_id')}' not found."}),
                None,
                False,
            )
        target = args.get("target_column", "")
        if not target:
            return json.dumps({"error": "target_column is required."}), None, False

        time_column = args.get("time_column") or None
        loaded = load_dataset(ds.file_path, ds.table_name)

        if target not in loaded.dataframe.columns:
            available = list(loaded.dataframe.columns)
            log.warning(
                "train_model | target=%r NOT FOUND in dataset=%s | available: %s",
                target, ds.name, available[:30],
            )
            return (
                json.dumps({
                    "error": (
                        f"Target column '{target}' does not exist in dataset '{ds.name}'. "
                        f"Available columns ({len(available)}): {available}. "
                        "Request Feature Engineering to create this column first "
                        "(e.g. create_lead_target for forecasting targets)."
                    )
                }),
                None,
                False,
            )
        if time_column and time_column not in loaded.dataframe.columns:
            available = list(loaded.dataframe.columns)
            log.warning(
                "train_model | time_column=%r NOT FOUND in dataset=%s",
                time_column, ds.name,
            )
            return (
                json.dumps({
                    "error": (
                        f"time_column '{time_column}' does not exist in dataset "
                        f"'{ds.name}'. Available: {available}."
                    )
                }),
                None,
                False,
            )

        class_weight = args.get("class_weight") or None
        if class_weight == "none":
            class_weight = None
        settings = TrainingSettings(
            target_column=target,
            test_size=float(args.get("test_size", 0.2)),
            random_state=int(args.get("random_state", 42)),
            time_column=time_column,
            class_weight=class_weight,
        )
        include_models = args.get("include_models") or None
        custom_models = args.get("custom_models") or None
        split_mode = "chronological" if time_column else "random"
        log.info(
            "train_model | dataset=%s target=%s split=%s time_column=%r test_size=%.2f include=%s custom=%s class_weight=%s",
            ds.name, target, split_mode, time_column, settings.test_size,
            include_models or "all", len(custom_models) if custom_models else 0, class_weight,
        )
        try:
            result = model = x_train_split = x_test_split = y_train_split = y_test_split = None
            for _event in train_automl_stream(
                loaded.dataframe, settings,
                custom_models=custom_models,
                include_models=include_models,
            ):
                if _event["type"] == "done":
                    result = _event["result"]
                    model = _event["pipeline"]
                    x_train_split = _event.get("x_train")
                    x_test_split = _event.get("x_test")
                    y_train_split = _event.get("y_train")
                    y_test_split = _event.get("y_test")
            if result is None:
                raise RuntimeError("Training stream ended without a result.")
        except Exception as exc:
            log.error("train_model | FAILED dataset=%s target=%s error=%s", ds.name, target, exc)
            return json.dumps({"error": f"Training failed: {exc}"}), None, False

        project = self._ctx.store.get_project(self._ctx.project_id)
        profile = profile_dataframe(loaded.dataframe)
        metadata = result.to_metadata()
        metadata["dataset"] = ds.to_dict()
        report_text = build_markdown_report(project.name, ds.to_dict(), metadata, profile)
        run_path = self._ctx.store.save_run(self._ctx.project_id, metadata, model, report_text)

        train_data_path: str | None = None
        test_data_path: str | None = None
        try:
            if x_train_split is not None and y_train_split is not None:
                train_df = x_train_split.copy()
                train_df[target] = y_train_split.values
                train_data_path = str(run_path / "train_data.csv")
                train_df.to_csv(train_data_path, index=False)
            if x_test_split is not None and y_test_split is not None:
                test_df = x_test_split.copy()
                test_df[target] = y_test_split.values
                test_data_path = str(run_path / "test_data.csv")
                test_df.to_csv(test_data_path, index=False)
        except Exception as exc:
            log.warning("train_model | failed to save split CSVs: %s", exc)

        summary = {
            "run_id": result.run_id,
            "dataset": ds.name,
            "dataset_id": ds.id,
            "target": target,
            "task_type": result.task_type,
            "best_model": result.best_model_name,
            "best_metrics": result.best_metrics,
            "split_mode": split_mode,
            "time_column": time_column,
            "model_path": str(run_path / "model.joblib"),
            "train_data_path": train_data_path,
            "test_data_path": test_data_path,
        }
        self._ctx.training_runs.append(summary)
        self._run_ids.append(result.run_id)

        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in result.best_metrics.items())
        log.info(
            "train_model | OK run_id=%s split=%s best_model=%s metrics=%s",
            result.run_id, split_mode, result.best_model_name, metrics_str,
        )

        diag_fig = None
        try:
            diag_fig = build_primary_diagnostic_figure(result.diagnostics, target_name=target)
        except Exception as exc:  # pragma: no cover
            log.warning("train_model | primary diagnostic failed: %s", exc)

        step_data: dict[str, Any] = dict(to_json_safe(summary))
        if diag_fig is not None:
            step_data["figure"] = diag_fig
        step = self._step(
            "training",
            f"Trained: {ds.name} → {target} ({split_mode} split)",
            f"Task: {result.task_type} | Best: {result.best_model_name} | {metrics_str}",
            data=step_data,
        )

        text_payload = json.dumps(to_json_safe({
            "run_id": result.run_id,
            "task_type": result.task_type,
            "best_model": result.best_model_name,
            "best_metrics": result.best_metrics,
            "split_mode": split_mode,
            "is_time_series": bool(result.diagnostics.get("is_time_series")),
            "n_test_points": result.diagnostics.get("n_points"),
        }))
        tool_content = vision_tool_content(text_payload, diag_fig)
        return tool_content, step, False

    def _create_model_chart(self, args: dict) -> tuple[str | list, AutopilotStep | None, bool]:
        run_id = args.get("run_id", "")
        chart_type = args.get("chart_type", "")
        top_n = int(args.get("top_n", 20))
        run = self._find_run(run_id)
        if run is None:
            log.warning("create_model_chart | run_id=%r not found", run_id)
            return json.dumps({"error": f"Run '{run_id}' not found."}), None, False

        target = run.get("target_column", "target")
        diag = run.get("diagnostics") or {}
        figs = build_diagnostic_figures(diag, target_name=target)
        figs_by_type = {
            "forecast": "Forecast vs Actual over Time",
            "predicted_vs_actual": "Predicted vs Actual",
            "residuals": "Residuals",
            "confusion_matrix": "Confusion Matrix",
        }

        fig = None
        title = ""
        if chart_type in figs_by_type:
            match_title = figs_by_type[chart_type]
            for t, f in figs:
                if t == match_title:
                    fig, title = f, f"{run_id} — {t}"
                    break
        elif chart_type == "residuals_over_time":
            fig = build_residuals_over_time_figure(diag, target_name=target)
            title = f"{run_id} — Residuals over Time"
        elif chart_type == "feature_importance":
            model_path = run.get("model_path")
            if not model_path:
                return json.dumps({"error": "Run has no saved model_path."}), None, False
            fig = build_feature_importance_figure(model_path, top_n=top_n, run_label=run_id)
            title = f"{run_id} — Feature Importance"
        elif chart_type == "leaderboard":
            fig = build_leaderboard_figure(
                run.get("leaderboard", []), run.get("task_type", ""), run_label=run_id,
            )
            title = f"{run_id} — Leaderboard"

        if fig is None:
            log.warning(
                "create_model_chart | could not build chart_type=%r for run=%s "
                "(task=%s is_ts=%s)", chart_type, run_id,
                run.get("task_type"), diag.get("is_time_series"),
            )
            return (
                json.dumps({
                    "error": (
                        f"Could not build '{chart_type}' for run '{run_id}'. "
                        f"Task type is '{run.get('task_type')}' "
                        f"(is_time_series={bool(diag.get('is_time_series'))}). "
                        "Some chart types only apply to certain task types or "
                        "tree-based models."
                    )
                }),
                None,
                False,
            )

        log.info("create_model_chart | OK run_id=%s chart_type=%s", run_id, chart_type)
        step = self._step(
            "chart",
            title,
            f"Diagnostic chart for {run_id}",
            data={"figure": fig, "run_id": run_id, "chart_type": chart_type},
        )
        text = json.dumps({"chart": title, "run_id": run_id, "chart_type": chart_type})
        return vision_tool_content(text, fig), step, False

    def _compare_runs(self, args: dict) -> tuple[str | list, AutopilotStep | None, bool]:
        run_ids = args.get("run_ids") or []
        metric = args.get("metric") or None
        if not run_ids or not isinstance(run_ids, list):
            return json.dumps({"error": "run_ids must be a non-empty list."}), None, False

        all_runs = self._ctx.store.list_runs(self._ctx.project_id)
        by_id = {r.get("run_id"): r for r in all_runs}
        chosen = [by_id[rid] for rid in run_ids if rid in by_id]
        missing = [rid for rid in run_ids if rid not in by_id]
        if not chosen:
            return (
                json.dumps({"error": f"None of the run_ids found. Missing: {missing}"}),
                None,
                False,
            )

        fig = build_run_comparison_figure(chosen, metric_key=metric)
        if fig is None:
            return (
                json.dumps({
                    "error": (
                        f"Could not build comparison — no shared metric across runs. "
                        f"Try metric='r2' or 'rmse' for regression, 'f1_weighted' "
                        f"for classification."
                    )
                }),
                None,
                False,
            )

        log.info("compare_runs | OK n=%d metric=%s missing=%s", len(chosen), metric, missing)
        step = self._step(
            "chart",
            f"Run comparison ({len(chosen)} runs)",
            f"Compared: {[r.get('run_id') for r in chosen]}",
            data={"figure": fig, "run_ids": run_ids},
        )
        text = json.dumps({
            "compared_run_ids": [r.get("run_id") for r in chosen],
            "missing_run_ids": missing,
            "metric": metric,
        })
        return vision_tool_content(text, fig), step, False

    def _build_ensemble(self, args: dict) -> tuple[str, AutopilotStep | None, bool]:
        from ..ingestion import load_dataset
        from ..training import (
            TrainingSettings, _candidate_models, build_preprocessor,
            infer_task_type, CLASSIFICATION,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import (
            VotingClassifier, VotingRegressor,
            StackingClassifier, StackingRegressor,
        )
        from sklearn.linear_model import LogisticRegression, Ridge

        ds = self._ctx.find_dataset(args.get("dataset_id", ""))
        if ds is None:
            return json.dumps({"error": f"Dataset '{args.get('dataset_id')}' not found."}), None, False
        target = args.get("target_column", "")
        model_names = args.get("model_names") or []
        ensemble_type = args.get("ensemble_type", "voting")
        time_column = args.get("time_column") or None
        class_weight = args.get("class_weight") or None
        if class_weight == "none":
            class_weight = None

        if len(model_names) < 2:
            return json.dumps({"error": "Provide at least 2 model_names for an ensemble."}), None, False

        loaded = load_dataset(ds.file_path, ds.table_name)
        df = loaded.dataframe.dropna(subset=[target])
        if target not in df.columns:
            return json.dumps({"error": f"Target '{target}' not found."}), None, False

        feature_cols = [c for c in df.columns if c != target and c != time_column]
        X = df[feature_cols]
        y = df[target]
        task = infer_task_type(y)

        candidates = _candidate_models(task, 42, 1, class_weight=class_weight)
        missing = [n for n in model_names if n not in candidates]
        if missing:
            return json.dumps({"error": f"Models not found: {missing}. Available: {list(candidates.keys())}"}), None, False

        preprocessor = build_preprocessor(X)
        estimators = []
        for name in model_names:
            inner_pipe = Pipeline([("preprocessor", preprocessor), ("model", candidates[name])])
            estimators.append((name.replace(" ", "_").lower(), inner_pipe))

        try:
            if ensemble_type == "stacking":
                meta = LogisticRegression(max_iter=500) if task == CLASSIFICATION else Ridge()
                if task == CLASSIFICATION:
                    ensemble = StackingClassifier(estimators=estimators, final_estimator=meta, passthrough=False, n_jobs=1)
                else:
                    ensemble = StackingRegressor(estimators=estimators, final_estimator=meta, passthrough=False, n_jobs=1)
            else:
                if task == CLASSIFICATION:
                    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
                else:
                    ensemble = VotingRegressor(estimators=estimators, n_jobs=1)

            settings = TrainingSettings(
                target_column=target,
                test_size=float(args.get("test_size", 0.2)),
                random_state=42,
                time_column=time_column,
                class_weight=class_weight,
            )
            # Train via train_automl using only the ensemble as a custom model
            custom_spec = [{"name": f"{ensemble_type.title()} Ensemble", "class": "__ensemble__"}]
            # We train it directly instead of through the registry
            from ..training import build_preprocessor as bp, _evaluate_model, _rank_leaderboard
            from sklearn.model_selection import train_test_split
            import pandas as pd

            if time_column and time_column in df.columns:
                from pandas import to_datetime
                order = to_datetime(df[time_column], errors="coerce").fillna(df[time_column])
                sort_idx = order.argsort(kind="mergesort")
                X_sorted = X.iloc[sort_idx].reset_index(drop=True)
                y_sorted = y.iloc[sort_idx].reset_index(drop=True)
                split_at = int(len(X_sorted) * (1 - settings.test_size))
                x_train, x_test = X_sorted.iloc[:split_at], X_sorted.iloc[split_at:]
                y_train, y_test = y_sorted.iloc[:split_at], y_sorted.iloc[split_at:]
            else:
                x_train, x_test, y_train, y_test = train_test_split(
                    X, y, test_size=settings.test_size, random_state=42
                )

            ensemble.fit(x_train, y_train)
            preds = ensemble.predict(x_test)
            metrics = _evaluate_model(task, y_test, preds, ensemble, x_test)
        except Exception as exc:
            log.error("build_ensemble | FAILED: %s", exc)
            return json.dumps({"error": f"Ensemble build failed: {exc}"}), None, False

        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        result = {
            "ensemble_type": ensemble_type,
            "task_type": task,
            "model_names": model_names,
            "metrics": to_json_safe(metrics),
            "summary": f"{ensemble_type.title()} ensemble of {model_names}: {metrics_str}",
        }
        log.info("build_ensemble | OK type=%s task=%s metrics=%s", ensemble_type, task, metrics_str)
        step = self._step(
            "observation",
            f"{ensemble_type.title()} Ensemble: {metrics_str}",
            f"Models: {model_names}",
            data=result,
        )
        return json.dumps(to_json_safe(result)), step, False

    def _find_run(self, run_id: str) -> dict | None:
        if not run_id:
            return None
        for r in self._ctx.store.list_runs(self._ctx.project_id):
            if r.get("run_id") == run_id:
                return r
        return None

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def _cross_validate(self, args: dict) -> tuple[str, AutopilotStep | None, bool]:
        from ..ingestion import load_dataset
        from ..training import (
            _candidate_models, build_preprocessor, infer_task_type, CLASSIFICATION,
        )
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold
        from sklearn.pipeline import Pipeline

        ds = self._ctx.find_dataset(args.get("dataset_id", ""))
        if ds is None:
            return json.dumps({"error": f"Dataset '{args.get('dataset_id')}' not found."}), None, False
        target = args.get("target_column", "")
        model_name = args.get("model_name", "")
        n_splits = int(args.get("n_splits") or 5)
        time_column = args.get("time_column") or None
        class_weight = args.get("class_weight", "none")

        loaded = load_dataset(ds.file_path, ds.table_name)
        df = loaded.dataframe.dropna(subset=[target])
        if target not in df.columns:
            return json.dumps({"error": f"Target '{target}' not found."}), None, False

        feature_cols = [c for c in df.columns if c != target and c != time_column]
        X = df[feature_cols]
        y = df[target]
        task = infer_task_type(y)

        candidates = _candidate_models(task, 42, 1)
        if model_name not in candidates:
            return json.dumps({"error": f"Model '{model_name}' not found. Available: {list(candidates.keys())}"}), None, False

        model = candidates[model_name]
        if class_weight == "balanced" and task == CLASSIFICATION and hasattr(model, "class_weight"):
            model.class_weight = "balanced"

        preprocessor = build_preprocessor(X)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        cv = (TimeSeriesSplit(n_splits=n_splits) if time_column
              else KFold(n_splits=n_splits, shuffle=True, random_state=42))
        scoring = "f1_weighted" if task == CLASSIFICATION else "r2"

        try:
            scores = cross_val_score(pipeline, X.fillna(X.median(numeric_only=True)), y,
                                     cv=cv, scoring=scoring, n_jobs=-1)
        except Exception as exc:
            return json.dumps({"error": f"Cross-validation failed: {exc}"}), None, False

        import numpy as np
        result = {
            "model": model_name, "task_type": task,
            "cv_type": "TimeSeriesSplit" if time_column else "KFold",
            "n_splits": n_splits, "scoring": scoring,
            "mean_score": round(float(scores.mean()), 4),
            "std_score": round(float(scores.std()), 4),
            "scores": [round(float(s), 4) for s in scores],
        }
        log.info("cross_validate | model=%s scoring=%s mean=%.4f std=%.4f",
                 model_name, scoring, scores.mean(), scores.std())
        step = self._step(
            "observation",
            f"CV: {model_name} — {scoring}={result['mean_score']:.4f} ± {result['std_score']:.4f}",
            json.dumps(result),
        )
        return json.dumps(to_json_safe(result)), step, False

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    _DEFAULT_PARAM_GRIDS: dict[str, dict] = {
        "Random Forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5],
        },
        "Extra Trees": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.05, 0.1, 0.2],
        },
        "Hist Gradient Boosting": {
            "model__max_iter": [100, 200, 300],
            "model__max_depth": [None, 5, 10],
            "model__learning_rate": [0.05, 0.1, 0.2],
        },
        "XGBoost": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 1.0],
        },
        "LightGBM": {
            "model__n_estimators": [100, 200, 300],
            "model__num_leaves": [20, 31, 50],
            "model__learning_rate": [0.03, 0.05, 0.1],
        },
        "Ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "Lasso": {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
        "ElasticNet": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.25, 0.5, 0.75],
        },
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__solver": ["lbfgs", "saga"],
        },
        "K-Nearest Neighbors": {"model__n_neighbors": [3, 5, 7, 11, 15]},
        "MLP": {
            "model__hidden_layer_sizes": [(50,), (100,), (100, 50), (200, 100)],
            "model__alpha": [0.0001, 0.001, 0.01],
        },
    }

    def _tune(self, args: dict) -> tuple[str, AutopilotStep | None, bool]:
        from ..ingestion import load_dataset
        from ..training import (
            _candidate_models, build_preprocessor, infer_task_type, CLASSIFICATION,
        )
        from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, KFold
        from sklearn.pipeline import Pipeline

        ds = self._ctx.find_dataset(args.get("dataset_id", ""))
        if ds is None:
            return json.dumps({"error": f"Dataset '{args.get('dataset_id')}' not found."}), None, False
        target = args.get("target_column", "")
        model_name = args.get("model_name", "")
        n_iter = int(args.get("n_iter") or 20)
        n_splits = int(args.get("n_splits") or 3)
        time_column = args.get("time_column") or None
        user_grid = args.get("param_grid") or {}

        loaded = load_dataset(ds.file_path, ds.table_name)
        df = loaded.dataframe.dropna(subset=[target])
        if target not in df.columns:
            return json.dumps({"error": f"Target '{target}' not found."}), None, False

        feature_cols = [c for c in df.columns if c != target and c != time_column]
        X = df[feature_cols]
        y = df[target]
        task = infer_task_type(y)

        candidates = _candidate_models(task, 42, 1)
        if model_name not in candidates:
            return json.dumps({"error": f"Model '{model_name}' not found. Available: {list(candidates.keys())}"}), None, False

        param_grid = user_grid or self._DEFAULT_PARAM_GRIDS.get(model_name, {})
        if not param_grid:
            return json.dumps({
                "error": (
                    f"No default parameter grid for '{model_name}'. "
                    "Pass params.param_grid explicitly, e.g. "
                    '{"model__n_estimators": [100, 200, 300], "model__max_depth": [3, 5, 7]}'
                )
            }), None, False

        preprocessor = build_preprocessor(X)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", candidates[model_name])])
        cv = (TimeSeriesSplit(n_splits=n_splits) if time_column
              else KFold(n_splits=n_splits, shuffle=True, random_state=42))
        scoring = "f1_weighted" if task == CLASSIFICATION else "r2"

        try:
            search = RandomizedSearchCV(
                pipeline, param_grid, n_iter=n_iter, cv=cv, scoring=scoring,
                random_state=42, n_jobs=-1, refit=True,
            )
            search.fit(X.fillna(X.median(numeric_only=True)), y)
        except Exception as exc:
            return json.dumps({"error": f"Hyperparameter search failed: {exc}"}), None, False

        best_params = {k: v for k, v in search.best_params_.items()}
        result = {
            "model": model_name, "task_type": task,
            "best_score": round(float(search.best_score_), 4),
            "scoring": scoring,
            "best_params": best_params,
            "n_iter": n_iter,
            "recommendation": (
                f"Best {scoring}={search.best_score_:.4f} with {best_params}. "
                "Pass these via custom_models in train_model to lock them in."
            ),
        }
        log.info("tune_hyperparameters | model=%s best_score=%.4f best_params=%s",
                 model_name, search.best_score_, best_params)
        step = self._step(
            "observation",
            f"Tuned: {model_name} — {scoring}={result['best_score']:.4f}",
            json.dumps(result),
        )
        return json.dumps(to_json_safe(result)), step, False
