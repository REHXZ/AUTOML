"""Model Tester Agent: evaluates trained models on held-out test data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd

from ..logging_setup import configure_logging
from .base import AgentContext, AutopilotStep, BaseAgent, to_json_safe

configure_logging()
log = logging.getLogger(__name__)


class ModelTesterAgent(BaseAgent):
    """Loads saved pipelines, runs them against held-out test CSVs, and reports metrics."""

    name = "model_tester"
    display_name = "Model Tester"

    def __init__(self, client, deployment: str, context: AgentContext) -> None:
        super().__init__(client, deployment, context)
        self._results: list[dict[str, Any]] = []

    def run(
        self, instructions: str
    ) -> Generator[AutopilotStep, list[str] | None, dict[str, Any]]:
        yield self._step(
            "agent_start",
            "Model Tester dispatched",
            instructions or "(evaluate trained models on held-out test data)",
        )

        runs_to_test = [r for r in self._ctx.training_runs if r.get("test_data_path")]
        if not runs_to_test:
            msg = (
                "No training runs with saved test data found. "
                "Ensure the Modeling Agent has run at least once."
            )
            log.warning("ModelTesterAgent | %s", msg)
            yield self._step("tool_result", "No test data available", msg)
            yield self._step("agent_end", "Model Tester finished", "")
            return {"error": msg, "test_results": []}

        for run_summary in runs_to_test:
            yield from self._evaluate_run(run_summary)

        if self._results:
            obs_text = _format_results_markdown(self._results)
            self._ctx.notebook.append(obs_text)
            yield self._step("observation", "Model Tester — held-out test results", obs_text)

        yield self._step("agent_end", "Model Tester finished", "")
        return {
            "test_results": [to_json_safe(r) for r in self._results],
            "runs_evaluated": len(self._results),
        }

    def _evaluate_run(
        self, run_summary: dict[str, Any]
    ) -> Generator[AutopilotStep, list[str] | None, None]:
        import joblib

        run_id = run_summary["run_id"]
        test_data_path = run_summary.get("test_data_path", "")
        target = run_summary.get("target", "")
        task_type = run_summary.get("task_type", "")
        best_model = run_summary.get("best_model", "unknown")

        yield self._step(
            "tool_call",
            f"Evaluating run {run_id}",
            f"target={target}  task={task_type}  model={best_model}",
        )

        model_path = run_summary.get("model_path") or str(
            self._ctx.store.project_path(self._ctx.project_id) / "runs" / run_id / "model.joblib"
        )

        if not Path(model_path).exists():
            msg = f"model.joblib not found at {model_path}"
            log.warning("ModelTesterAgent | run=%s %s", run_id, msg)
            yield self._step("tool_result", f"Run {run_id} — model missing", msg)
            return

        if not Path(test_data_path).exists():
            msg = f"test_data.csv not found at {test_data_path}"
            log.warning("ModelTesterAgent | run=%s %s", run_id, msg)
            yield self._step("tool_result", f"Run {run_id} — test data missing", msg)
            return

        try:
            pipeline = joblib.load(model_path)
            test_df = pd.read_csv(test_data_path)
        except Exception as exc:
            msg = f"Failed to load artifacts for run {run_id}: {exc}"
            log.error("ModelTesterAgent | %s", msg)
            yield self._step("tool_result", f"Run {run_id} — load error", msg)
            return

        if target not in test_df.columns:
            msg = f"Target column '{target}' not found in test_data.csv"
            log.warning("ModelTesterAgent | run=%s %s", run_id, msg)
            yield self._step("tool_result", f"Run {run_id} — target missing", msg)
            return

        x_test = test_df.drop(columns=[target])
        y_test = test_df[target]

        try:
            y_pred = pipeline.predict(x_test)
            test_metrics = _compute_metrics(task_type, y_test, y_pred, pipeline, x_test)
        except Exception as exc:
            msg = f"Prediction/evaluation failed for run {run_id}: {exc}"
            log.error("ModelTesterAgent | %s", msg)
            yield self._step("tool_result", f"Run {run_id} — prediction error", msg)
            return

        result: dict[str, Any] = {
            "run_id": run_id,
            "task_type": task_type,
            "best_model": best_model,
            "test_size": int(len(y_test)),
            "test_metrics": test_metrics,
        }
        self._results.append(result)

        metrics_str = ", ".join(
            f"{k}: {v:.4f}" for k, v in test_metrics.items() if isinstance(v, float)
        )
        log.info("ModelTesterAgent | run=%s metrics=%s", run_id, metrics_str)
        yield self._step(
            "tool_result",
            f"Run {run_id} — test evaluation complete",
            metrics_str,
            data=to_json_safe(result),
        )


def _compute_metrics(
    task_type: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
    pipeline: Any,
    x_test: pd.DataFrame,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
    )

    metrics: dict[str, float] = {}

    if task_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["f1_weighted"] = float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        metrics["precision_weighted"] = float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        metrics["recall_weighted"] = float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        classes = np.unique(y_test)
        if len(classes) == 2 and hasattr(pipeline, "predict_proba"):
            try:
                from sklearn.metrics import roc_auc_score

                proba = pipeline.predict_proba(x_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
            except Exception:
                pass
    else:
        metrics["r2"] = float(r2_score(y_test, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        try:
            from sklearn.metrics import median_absolute_error

            metrics["median_ae"] = float(median_absolute_error(y_test, y_pred))
        except Exception:
            pass

    return metrics


def _format_results_markdown(results: list[dict[str, Any]]) -> str:
    lines = ["## Model Test Results (Held-Out Data)\n"]
    for r in results:
        lines.append(f"### Run `{r['run_id']}` — {r['task_type'].title()}")
        lines.append(
            f"**Model:** {r.get('best_model', 'unknown')}  |  **Test rows:** {r['test_size']}\n"
        )
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in r["test_metrics"].items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
        lines.append("")
    return "\n".join(lines)
