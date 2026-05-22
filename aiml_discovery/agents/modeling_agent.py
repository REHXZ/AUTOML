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

        settings = TrainingSettings(
            target_column=target,
            test_size=float(args.get("test_size", 0.2)),
            random_state=int(args.get("random_state", 42)),
            time_column=time_column,
        )
        include_models = args.get("include_models") or None
        custom_models = args.get("custom_models") or None
        split_mode = "chronological" if time_column else "random"
        log.info(
            "train_model | dataset=%s target=%s split=%s time_column=%r test_size=%.2f include=%s custom=%s",
            ds.name, target, split_mode, time_column, settings.test_size,
            include_models or "all", len(custom_models) if custom_models else 0,
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

    def _find_run(self, run_id: str) -> dict | None:
        if not run_id:
            return None
        for r in self._ctx.store.list_runs(self._ctx.project_id):
            if r.get("run_id") == run_id:
                return r
        return None
