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
from ..training import TrainingSettings, train_automl
from .base import AgentContext, AutopilotStep, BaseAgent, to_json_safe, vision_tool_content

configure_logging()
log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are the Modeling Agent — the team's hands-on ML engineer. You have
broad authority to train, evaluate, and visualise as you see fit. The
Scientist gives you a high-level objective; YOU choose the experiments
that best answer it.

────────────────────────────────────────────────────────────────────────
ALWAYS START WITH inspect_dataset
  Confirm the target column exists. Note the time column (look for
  datetime-like names: request_month, order_entry_date, ds, date, etc.)
  and any lead/lag/rolling columns the FE Agent created.

PICK THE RIGHT TRAINING MODE
  • For FORECASTING / TIME-SERIES problems → call train_model with
    time_column set to the date column. This forces a CHRONOLOGICAL
    holdout: the last test_size fraction of rows in time order becomes
    the test set. This is the ONLY honest backtest for forecasting.
    Train on lead targets like qty_lead_1, qty_lead_3 that FE created.
  • For non-temporal problems → omit time_column (random split).

  If the dataset has a time-like column but no lead/lag features and the
  Scientist's objective is forecasting next-period values, you should
  STOP and record_finding that FE needs to run
  create_lead_target / create_lag_features / create_rolling_features
  first. Then done() with rationale="BLOCKED: forecasting features
  needed". Do NOT pretend a same-row regression on raw qty is a forecast.

EXPERIMENT FREELY
  • Try multiple targets (e.g. qty_lead_1 and qty_lead_3 separately).
  • Try multiple test_size values (0.2 for long series, 0.3 for short).
  • Re-train with/without specific features by routing back to FE if
    needed.

VISUALISE WITH create_model_chart
  After every train_model you automatically receive the primary
  diagnostic chart. You can also call create_model_chart for ANY past
  run_id to render extra perspectives:
    • predicted_vs_actual   – regression scatter with y=x reference
    • forecast              – actual vs predicted lines aligned in time
    • residuals             – residual scatter (regression)
    • residuals_over_time   – residuals plotted in time order
    • confusion_matrix      – classification
    • feature_importance    – top-N importances (tree models only)
    • leaderboard           – primary metric per candidate model in run
  Use compare_runs with a list of run_ids to put the primary metric
  side-by-side across experiments.

SAY WHAT YOU SEE
  For every chart, write a short observation: are predictions tracking
  the actuals, or flat at the mean? Do residuals drift over time
  (concept drift)? Is one feature dominating importance (possible
  leakage)? Tie every recommendation to something visible in a chart.

FINISH WITH done(summary)
  • Strongest run_id (named honestly — usually highest R²/F1 or
    lowest RMSE on the chronological holdout, NOT the random-split one)
  • rationale citing concrete metrics
  • concerns array if you suspect leakage or unstable training

AVAILABLE MODELS (pass names verbatim in include_models to select a subset)
  Classification:
    Baseline (Majority), Logistic Regression, SGD Classifier, Linear SVC,
    Gaussian Naive Bayes, Bernoulli Naive Bayes, Linear Discriminant Analysis,
    Quadratic Discriminant Analysis, Decision Tree, Extra Trees,
    Random Forest, AdaBoost, Gradient Boosting, Hist Gradient Boosting,
    K-Nearest Neighbors, SVC (RBF), MLP, Bagging,
    XGBoost*, LightGBM*, CatBoost*          (* installed automatically if present)

  Regression:
    Baseline (Mean), Linear Regression, Ridge, Lasso, ElasticNet,
    Bayesian Ridge, Huber Regressor, SGD Regressor, Decision Tree,
    Extra Trees, Random Forest, AdaBoost, Gradient Boosting,
    Hist Gradient Boosting, K-Nearest Neighbors, Linear SVR, SVR (RBF),
    MLP, Bagging,
    XGBoost*, LightGBM*, CatBoost*

  Speed guide (approximate, scales with dataset size):
    Fast   → Baseline, Linear*, Lasso, Ridge, ElasticNet, Bayesian Ridge,
              Naive Bayes, LDA, Decision Tree, Hist Gradient Boosting,
              SGD*, Linear SVR / Linear SVC
    Medium → Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
              K-Nearest Neighbors, XGBoost, LightGBM, CatBoost, Bagging
    Slow   → SVR (RBF), SVC (RBF), MLP  [avoid on > 50k rows]
    QDA    → can fail with many features / small classes

  For large datasets (> 50k rows), pass include_models to skip slow models.
  For small datasets (< 5k rows), all models are safe.

CUSTOM MODELS
  Pass custom_models as a list of specs:
    [{"name": "My XGBoost", "class": "xgboost.XGBRegressor",
      "params": {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 5}}]
  Any sklearn-compatible estimator is valid. The class must be importable.
  Custom models run AFTER standard ones and are always included.

CROSS-VALIDATION (cross_validate_model)
  Use when:
    – The dataset is small (< 5k rows) and you want a reliable estimate
      before committing to a full train_model run.
    – You want to compare two models on the same fold structure.
    – You need to check if a model degrades badly across folds (high std).
  Set time_column to get TimeSeriesSplit; omit for KFold.
  Set class_weight="balanced" for imbalanced classification.

HYPERPARAMETER TUNING (tune_hyperparameters)
  Use AFTER you have identified the best model family from the baseline
  leaderboard. Runs RandomizedSearchCV with a sensible default param grid.
  The result includes the best params; pass them via custom_models in a
  follow-up train_model to lock them in as a tuned model.
  Avoid tuning slow models (SVR, SVC, MLP) on large datasets.

CLASS IMBALANCE
  If EDA reports class imbalance (ratio > 3×), consider:
    1. Feature Engineering → smote (oversample the minority class first), OR
    2. cross_validate_model / train_model with class_weight="balanced"
       (cost-sensitive training; no extra rows needed).
  For very severe imbalance (ratio > 10×), prefer SMOTE + class_weight together.

USE THE RESEARCHER WHEN YOU NEED EXTERNAL KNOWLEDGE
  Call spawn_researcher if you need to:
    – Look up best-known benchmarks for the task type / dataset size.
    – Verify whether a particular technique or hyperparameter choice
      is appropriate for the problem domain.
    – Investigate an unusual pattern you see in the diagnostics.
  Pass a specific, focused research question.

You have authority to call charts and retrain. You do NOT have
authority to fabricate metrics — if something fails, say so plainly
and propose the fix.
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
                "name": "train_arima",
                "description": (
                    "Fit an ARIMA or SARIMA model to a univariate time series. "
                    "Automatically selects the best (p,d,q) order via AIC grid search "
                    "(uses pmdarima if installed, otherwise statsmodels ARIMA). "
                    "Returns model fit metrics and in-sample residual stats. "
                    "Use train_model with time_column for multi-feature forecasting; "
                    "use train_arima for pure univariate TS benchmarks."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "target_column": {
                            "type": "string",
                            "description": "The univariate series to model.",
                        },
                        "time_column": {
                            "type": "string",
                            "description": "Optional column to sort by before fitting.",
                        },
                        "seasonal": {
                            "type": "boolean",
                            "description": "If true, fit SARIMA with seasonal period m. Default false.",
                        },
                        "m": {
                            "type": "integer",
                            "description": "Seasonal period (e.g. 12 for monthly, 7 for daily). Default 12.",
                        },
                        "max_p": {"type": "integer", "description": "Max AR order. Default 3."},
                        "max_q": {"type": "integer", "description": "Max MA order. Default 3."},
                        "max_d": {"type": "integer", "description": "Max integration order. Default 2."},
                    },
                    "required": ["dataset_id", "target_column"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "arima_forecast",
                "description": (
                    "Produce n-step-ahead out-of-sample forecasts from a previously fitted ARIMA model. "
                    "Returns point forecasts and 95% confidence bands."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arima_run_id": {
                            "type": "string",
                            "description": "The run_id returned by train_arima.",
                        },
                        "steps": {
                            "type": "integer",
                            "description": "Number of periods to forecast ahead. Default 12.",
                        },
                    },
                    "required": ["arima_run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "explain_model",
                "description": (
                    "Compute SHAP Shapley values for a saved run and return a feature-importance "
                    "summary plot (base64 PNG) plus a table of mean |SHAP| per feature. "
                    "Works with tree-based models (TreeExplainer, fast) and linear models "
                    "(LinearExplainer). Falls back to KernelExplainer for others (slow — "
                    "use a small sample_size). "
                    "Requires: pip install shap"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "sample_size": {
                            "type": "integer",
                            "description": "Rows to explain (default 200, max 1000). Larger → slower.",
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "Top-N features to show in the summary (default 15).",
                        },
                    },
                    "required": ["run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tune_hyperparameters_optuna",
                "description": (
                    "Bayesian hyperparameter optimisation via Optuna. Significantly more efficient "
                    "than random search when n_trials is limited. "
                    "Falls back to tune_hyperparameters (RandomizedSearchCV) if optuna is not installed. "
                    "Returns best params and their cross-validated score."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "target_column": {"type": "string"},
                        "model_name": {
                            "type": "string",
                            "description": "Name of the model to tune (e.g. 'Random Forest', 'XGBoost').",
                        },
                        "n_trials": {
                            "type": "integer",
                            "description": "Number of Optuna trials (default 30).",
                        },
                        "n_splits": {
                            "type": "integer",
                            "description": "CV folds (default 3).",
                        },
                        "time_column": {
                            "type": "string",
                            "description": "If set, uses TimeSeriesSplit.",
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
        if name == "train_arima":
            return self._train_arima(args)
        if name == "arima_forecast":
            return self._arima_forecast(args)
        if name == "explain_model":
            return self._explain_model(args)
        if name == "tune_hyperparameters_optuna":
            return self._tune_optuna(args)
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
            result, model = train_automl(
                loaded.dataframe, settings,
                custom_models=custom_models,
                include_models=include_models,
            )
        except Exception as exc:
            log.error("train_model | FAILED dataset=%s target=%s error=%s", ds.name, target, exc)
            return json.dumps({"error": f"Training failed: {exc}"}), None, False

        project = self._ctx.store.get_project(self._ctx.project_id)
        profile = profile_dataframe(loaded.dataframe)
        metadata = result.to_metadata()
        metadata["dataset"] = ds.to_dict()
        report_text = build_markdown_report(project.name, ds.to_dict(), metadata, profile)
        self._ctx.store.save_run(self._ctx.project_id, metadata, model, report_text)

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
            infer_task_type, CLASSIFICATION, train_automl,
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
    # ARIMA / SARIMA
    # ------------------------------------------------------------------

    # Stores fitted ARIMA models by run_id so arima_forecast can retrieve them.
    _arima_cache: dict[str, Any] = {}

    def _train_arima(self, args: dict) -> tuple[str, Any, bool]:
        ds = self._ctx.find_dataset(args.get("dataset_id", ""))
        if ds is None:
            return json.dumps({"error": f"Dataset '{args.get('dataset_id')}' not found."}), None, False

        target = args.get("target_column", "")
        time_col = args.get("time_column") or None
        seasonal = bool(args.get("seasonal", False))
        m = int(args.get("m", 12))
        max_p = int(args.get("max_p", 3))
        max_q = int(args.get("max_q", 3))
        max_d = int(args.get("max_d", 2))

        loaded = load_dataset(ds.file_path, ds.table_name)
        df = loaded.dataframe.copy()
        if target not in df.columns:
            return json.dumps({"error": f"Column '{target}' not found."}), None, False
        if time_col and time_col in df.columns:
            df = df.sort_values(time_col)
        series = df[target].dropna()
        if len(series) < 8:
            return json.dumps({"error": "Need ≥8 observations for ARIMA."}), None, False

        # Try pmdarima (auto_arima) first; fall back to statsmodels grid search
        fitted_model = None
        order = None
        seasonal_order = None
        aic = None
        try:
            import pmdarima as pm
            auto = pm.auto_arima(
                series,
                start_p=0, max_p=max_p,
                start_q=0, max_q=max_q,
                d=None, max_d=max_d,
                seasonal=seasonal, m=m,
                information_criterion="aic",
                suppress_warnings=True, error_action="ignore",
            )
            fitted_model = auto
            order = auto.order
            seasonal_order = auto.seasonal_order if seasonal else None
            aic = float(auto.aic())
            log.info("train_arima | pmdarima auto_arima | order=%s seasonal=%s aic=%.2f", order, seasonal_order, aic)
        except ImportError:
            # Fallback: statsmodels ARIMA grid search
            try:
                from statsmodels.tsa.arima.model import ARIMA as _ARIMA
                import itertools
                best_aic = float("inf")
                best_order = (1, 1, 1)
                for p, d, q in itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
                    if p + d + q == 0:
                        continue
                    try:
                        m_fit = _ARIMA(series, order=(p, d, q)).fit()
                        if m_fit.aic < best_aic:
                            best_aic = m_fit.aic
                            best_order = (p, d, q)
                            fitted_model = m_fit
                    except Exception:
                        continue
                order = best_order
                aic = best_aic
                log.info("train_arima | statsmodels grid | best_order=%s aic=%.2f", order, aic)
            except Exception as exc:
                return json.dumps({"error": f"ARIMA fitting failed: {exc}"}), None, False
        except Exception as exc:
            return json.dumps({"error": f"ARIMA fitting failed: {exc}"}), None, False

        if fitted_model is None:
            return json.dumps({"error": "Could not fit any ARIMA model."}), None, False

        import numpy as np
        try:
            if hasattr(fitted_model, "resid"):
                resid = np.array(fitted_model.resid()).flatten() if callable(fitted_model.resid) else np.array(fitted_model.resid).flatten()
            else:
                resid = np.array([])
        except Exception:
            resid = np.array([])

        import datetime
        run_id = f"arima_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        ModelingAgent._arima_cache[run_id] = {
            "model": fitted_model,
            "series": series,
            "target": target,
        }
        self._run_ids.append(run_id)

        rmse = float(np.sqrt(np.mean(resid ** 2))) if len(resid) > 0 else None
        result = {
            "run_id": run_id,
            "model_type": "SARIMA" if seasonal else "ARIMA",
            "target_column": target,
            "order": list(order),
            "seasonal_order": list(seasonal_order) if seasonal_order else None,
            "aic": round(aic, 4) if aic else None,
            "in_sample_rmse": round(rmse, 4) if rmse else None,
            "n_obs": int(len(series)),
        }
        log.info("train_arima | OK run_id=%s order=%s aic=%.4f", run_id, order, aic or 0)
        step = self._step(
            "training",
            f"ARIMA: {target} order={order}",
            f"AIC={aic:.2f} | in-sample RMSE={rmse:.4f}" if rmse else f"AIC={aic:.2f}",
            data=to_json_safe(result),
        )
        return json.dumps(to_json_safe(result)), step, False

    def _arima_forecast(self, args: dict) -> tuple[str, Any, bool]:
        arima_run_id = args.get("arima_run_id", "")
        steps = int(args.get("steps", 12))
        cached = ModelingAgent._arima_cache.get(arima_run_id)
        if cached is None:
            return json.dumps({"error": f"ARIMA run '{arima_run_id}' not found. Call train_arima first."}), None, False

        fitted_model = cached["model"]
        target = cached["target"]
        try:
            if hasattr(fitted_model, "predict"):
                # pmdarima-style
                try:
                    fc, conf_int = fitted_model.predict(n_periods=steps, return_conf_int=True, alpha=0.05)
                    lower = conf_int[:, 0].tolist()
                    upper = conf_int[:, 1].tolist()
                    fc_list = fc.tolist()
                except TypeError:
                    fc_res = fitted_model.get_forecast(steps=steps)
                    fc_list = fc_res.predicted_mean.tolist()
                    ci = fc_res.conf_int(alpha=0.05)
                    lower = ci.iloc[:, 0].tolist()
                    upper = ci.iloc[:, 1].tolist()
            else:
                fc_res = fitted_model.get_forecast(steps=steps)
                fc_list = fc_res.predicted_mean.tolist()
                ci = fc_res.conf_int(alpha=0.05)
                lower = ci.iloc[:, 0].tolist()
                upper = ci.iloc[:, 1].tolist()
        except Exception as exc:
            return json.dumps({"error": f"Forecast failed: {exc}"}), None, False

        import plotly.graph_objects as _go
        series = cached["series"]
        x_hist = list(range(len(series)))
        x_fc = list(range(len(series), len(series) + steps))
        fig = _go.Figure()
        fig.add_trace(_go.Scatter(x=x_hist, y=series.values.tolist(), mode="lines", name="Historical"))
        fig.add_trace(_go.Scatter(x=x_fc, y=fc_list, mode="lines", name="Forecast", line=dict(dash="dash")))
        fig.add_trace(_go.Scatter(
            x=x_fc + x_fc[::-1],
            y=upper + lower[::-1],
            fill="toself", fillcolor="rgba(0,100,255,0.15)",
            line=dict(color="rgba(255,255,255,0)"), name="95% CI",
        ))
        fig.update_layout(title=f"ARIMA Forecast: {target} ({steps} steps)", template="plotly_white")

        result = {
            "arima_run_id": arima_run_id,
            "target_column": target,
            "steps": steps,
            "forecast": [round(v, 4) for v in fc_list],
            "lower_95": [round(v, 4) for v in lower],
            "upper_95": [round(v, 4) for v in upper],
        }
        step = self._step(
            "chart",
            f"ARIMA Forecast: {target} ({steps} steps)",
            f"Point forecasts with 95% confidence bands",
            data={"figure": fig, **to_json_safe(result)},
        )
        return vision_tool_content(json.dumps(to_json_safe(result)), fig), step, False

    # ------------------------------------------------------------------
    # SHAP Explainability
    # ------------------------------------------------------------------

    def _explain_model(self, args: dict) -> tuple[str | list, Any, bool]:
        run_id = args.get("run_id", "")
        sample_size = min(int(args.get("sample_size", 200)), 1000)
        top_n = int(args.get("top_n", 15))

        run = self._find_run(run_id)
        if run is None:
            return json.dumps({"error": f"Run '{run_id}' not found."}), None, False
        model_path = run.get("model_path")
        if not model_path:
            return json.dumps({"error": "Run has no saved model_path."}), None, False

        ds_info = self._ctx.find_dataset(run.get("dataset_id", ""))
        if ds_info is None:
            ds_id = run.get("dataset", {}).get("id", "")
            ds_info = self._ctx.find_dataset(ds_id)
        if ds_info is None:
            return json.dumps({"error": "Could not locate the original dataset for this run."}), None, False

        target = run.get("target_column", "")
        time_col = run.get("time_column") or run.get("settings", {}).get("time_column")

        try:
            import shap
        except ImportError:
            return json.dumps({
                "error": "SHAP is not installed. Run: pip install shap"
            }), None, False

        import joblib
        import numpy as np

        try:
            pipeline = joblib.load(model_path)
        except Exception as exc:
            return json.dumps({"error": f"Could not load model: {exc}"}), None, False

        loaded = load_dataset(ds_info.file_path, ds_info.table_name)
        df = loaded.dataframe.copy()
        feature_cols = [c for c in df.columns if c != target and c != time_col]
        X = df[feature_cols].head(sample_size)

        try:
            preprocessor = pipeline.named_steps.get("preprocessor")
            model_step = pipeline.named_steps.get("model")
            if preprocessor is not None and model_step is not None:
                X_transformed = preprocessor.transform(X)
                explainer_target = model_step
            else:
                X_transformed = X
                explainer_target = pipeline

            model_type = type(model_step or explainer_target).__name__
            tree_types = {"RandomForestClassifier", "RandomForestRegressor",
                          "ExtraTreesClassifier", "ExtraTreesRegressor",
                          "GradientBoostingClassifier", "GradientBoostingRegressor",
                          "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
                          "DecisionTreeClassifier", "DecisionTreeRegressor",
                          "XGBClassifier", "XGBRegressor",
                          "LGBMClassifier", "LGBMRegressor",
                          "CatBoostClassifier", "CatBoostRegressor"}
            linear_types = {"LogisticRegression", "LinearRegression", "Ridge", "Lasso",
                             "ElasticNet", "SGDClassifier", "SGDRegressor", "LinearSVC", "LinearSVR"}

            if model_type in tree_types:
                explainer = shap.TreeExplainer(explainer_target)
                shap_values = explainer.shap_values(X_transformed)
            elif model_type in linear_types:
                explainer = shap.LinearExplainer(explainer_target, X_transformed)
                shap_values = explainer.shap_values(X_transformed)
            else:
                bg = shap.sample(X_transformed, min(50, len(X_transformed)))
                explainer = shap.KernelExplainer(explainer_target.predict, bg)
                shap_values = explainer.shap_values(X_transformed, nsamples=50)

            if isinstance(shap_values, list):
                sv = np.abs(shap_values[1]) if len(shap_values) > 1 else np.abs(shap_values[0])
            else:
                sv = np.abs(shap_values)

            if preprocessor is not None:
                try:
                    feature_names = preprocessor.get_feature_names_out().tolist()
                except Exception:
                    feature_names = [f"f{i}" for i in range(sv.shape[1])]
            else:
                feature_names = feature_cols

            mean_abs = sv.mean(axis=0)
            ranked = sorted(zip(feature_names, mean_abs.tolist()), key=lambda x: x[1], reverse=True)[:top_n]
        except Exception as exc:
            return json.dumps({"error": f"SHAP computation failed: {exc}"}), None, False

        import plotly.graph_objects as _go
        feat_names = [r[0] for r in ranked]
        feat_vals = [round(r[1], 4) for r in ranked]
        fig = _go.Figure(_go.Bar(
            x=feat_vals[::-1], y=feat_names[::-1], orientation="h",
            marker_color="steelblue",
        ))
        fig.update_layout(
            title=f"SHAP Feature Importance — {run_id} (top {top_n})",
            xaxis_title="Mean |SHAP value|",
            template="plotly_white",
        )

        result = {
            "run_id": run_id,
            "model_type": model_type,
            "sample_size": sample_size,
            "top_features": [{"feature": n, "mean_abs_shap": v} for n, v in ranked],
        }
        step = self._step(
            "chart",
            f"SHAP Importance — {run_id}",
            f"Top {top_n} features by mean |SHAP| ({model_type})",
            data={"figure": fig, **to_json_safe(result)},
        )
        return vision_tool_content(json.dumps(to_json_safe(result)), fig), step, False

    # ------------------------------------------------------------------
    # Bayesian HPO via Optuna
    # ------------------------------------------------------------------

    def _tune_optuna(self, args: dict) -> tuple[str, Any, bool]:
        ds = self._ctx.find_dataset(args.get("dataset_id", ""))
        if ds is None:
            return json.dumps({"error": f"Dataset '{args.get('dataset_id')}' not found."}), None, False
        target = args.get("target_column", "")
        model_name = args.get("model_name", "")
        n_trials = int(args.get("n_trials") or 30)
        n_splits = int(args.get("n_splits") or 3)
        time_column = args.get("time_column") or None

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            log.info("Optuna not installed — falling back to RandomizedSearchCV")
            fallback_args = dict(args)
            fallback_args["n_iter"] = n_trials
            result_str, step, term = self._tune(fallback_args)
            result = json.loads(result_str)
            result["note"] = "optuna not installed; used RandomizedSearchCV fallback"
            return json.dumps(to_json_safe(result)), step, term

        from ..ingestion import load_dataset as _ld
        from ..training import _candidate_models, build_preprocessor, infer_task_type, CLASSIFICATION
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold
        from sklearn.pipeline import Pipeline

        loaded = _ld(ds.file_path, ds.table_name)
        df = loaded.dataframe.dropna(subset=[target])
        if target not in df.columns:
            return json.dumps({"error": f"Target '{target}' not found."}), None, False

        feature_cols = [c for c in df.columns if c != target and c != time_column]
        X = df[feature_cols].fillna(df[feature_cols].select_dtypes("number").median())
        y = df[target]
        task = infer_task_type(y)
        scoring = "f1_weighted" if task == CLASSIFICATION else "r2"
        cv = (TimeSeriesSplit(n_splits=n_splits) if time_column
              else KFold(n_splits=n_splits, shuffle=True, random_state=42))

        candidates = _candidate_models(task, 42, 1)
        if model_name not in candidates:
            return json.dumps({"error": f"Model '{model_name}' not found. Available: {list(candidates.keys())}"}), None, False

        _OPTUNA_SPACES: dict[str, Any] = {
            "Random Forest": lambda t: {"model__n_estimators": t.suggest_int("n_est", 50, 400), "model__max_depth": t.suggest_categorical("max_d", [None, 5, 10, 20]), "model__min_samples_leaf": t.suggest_int("msl", 1, 10)},
            "Extra Trees": lambda t: {"model__n_estimators": t.suggest_int("n_est", 50, 400), "model__max_depth": t.suggest_categorical("max_d", [None, 5, 10, 20])},
            "Gradient Boosting": lambda t: {"model__n_estimators": t.suggest_int("n_est", 50, 400), "model__learning_rate": t.suggest_float("lr", 0.01, 0.3, log=True), "model__max_depth": t.suggest_int("max_d", 2, 8)},
            "Hist Gradient Boosting": lambda t: {"model__max_iter": t.suggest_int("n_est", 50, 400), "model__learning_rate": t.suggest_float("lr", 0.01, 0.3, log=True)},
            "XGBoost": lambda t: {"model__n_estimators": t.suggest_int("n_est", 50, 400), "model__learning_rate": t.suggest_float("lr", 0.01, 0.3, log=True), "model__max_depth": t.suggest_int("max_d", 2, 8), "model__subsample": t.suggest_float("ss", 0.5, 1.0)},
            "LightGBM": lambda t: {"model__n_estimators": t.suggest_int("n_est", 50, 400), "model__learning_rate": t.suggest_float("lr", 0.01, 0.3, log=True), "model__num_leaves": t.suggest_int("nl", 10, 100)},
            "Ridge": lambda t: {"model__alpha": t.suggest_float("alpha", 0.001, 100.0, log=True)},
            "Lasso": lambda t: {"model__alpha": t.suggest_float("alpha", 0.001, 10.0, log=True)},
            "Logistic Regression": lambda t: {"model__C": t.suggest_float("C", 0.001, 100.0, log=True)},
            "K-Nearest Neighbors": lambda t: {"model__n_neighbors": t.suggest_int("k", 2, 20)},
        }

        suggest_fn = _OPTUNA_SPACES.get(model_name)
        if suggest_fn is None:
            log.info("tune_optuna | no Optuna space for %s — falling back to RandomizedSearchCV", model_name)
            fallback_args = dict(args)
            fallback_args["n_iter"] = n_trials
            result_str, step, term = self._tune(fallback_args)
            result = json.loads(result_str)
            result["note"] = f"No Optuna space defined for '{model_name}'; used RandomizedSearchCV."
            return json.dumps(to_json_safe(result)), step, term

        preprocessor = build_preprocessor(X)

        def objective(trial):
            params = suggest_fn(trial)
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", candidates[model_name]),
            ])
            pipe.set_params(**params)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return float(scores.mean())

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_score = round(study.best_value, 4)
        log.info("tune_optuna | model=%s best_%s=%.4f params=%s", model_name, scoring, best_score, best_params)
        result = {
            "method": "Optuna Bayesian HPO",
            "model": model_name,
            "task_type": task,
            "n_trials": n_trials,
            "scoring": scoring,
            "best_score": best_score,
            "best_params": best_params,
            "recommendation": (
                f"Best {scoring}={best_score:.4f} with {best_params}. "
                "Pass these via custom_models in train_model to lock them in."
            ),
        }
        step = self._step(
            "observation",
            f"Optuna HPO: {model_name} — {scoring}={best_score:.4f}",
            json.dumps(result),
        )
        return json.dumps(to_json_safe(result)), step, False

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
