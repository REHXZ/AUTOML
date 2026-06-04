"""Diagnostic figures for trained models."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from backend.server.logging_setup import configure_logging

configure_logging()
log = logging.getLogger(__name__)

_MAX_POINTS = 5000  # cap diagnostic series to keep metadata JSON small


# ─────────────────────────────────────────────────────────────────────────────
# Build the diagnostics dict that gets saved inside TrainingResult
# ─────────────────────────────────────────────────────────────────────────────


def build_diagnostics_dict(
    task_type: str,
    y_test: pd.Series,
    y_pred: np.ndarray | list,
    x_test: pd.DataFrame,
    time_values: pd.Series | None = None,
) -> dict[str, Any]:
    if time_values is not None:
        time_labels, sort_idx = _time_axis_from_series(time_values)
    else:
        time_labels, sort_idx = _detect_time_axis(x_test)
    is_ts = time_labels is not None

    y_test_arr = pd.Series(y_test).reset_index(drop=True)
    y_pred_arr = pd.Series(np.asarray(y_pred)).reset_index(drop=True)

    if sort_idx is not None:
        order = np.argsort(sort_idx.reset_index(drop=True).values)
        y_test_arr = y_test_arr.iloc[order].reset_index(drop=True)
        y_pred_arr = y_pred_arr.iloc[order].reset_index(drop=True)
        time_labels = [time_labels[i] for i in order]

    n = len(y_test_arr)
    if n > _MAX_POINTS:
        step = max(1, n // _MAX_POINTS)
        y_test_arr = y_test_arr.iloc[::step].reset_index(drop=True)
        y_pred_arr = y_pred_arr.iloc[::step].reset_index(drop=True)
        if time_labels is not None:
            time_labels = time_labels[::step]

    diag = {
        "task_type": task_type,
        "is_time_series": is_ts,
        "y_test": _to_jsonable_list(y_test_arr),
        "y_pred": _to_jsonable_list(y_pred_arr),
        "time_index": time_labels,
        "n_points": len(y_test_arr),
    }
    log.debug(
        "build_diagnostics_dict | task=%s n=%d is_ts=%s",
        task_type, len(y_test_arr), is_ts,
    )
    return diag


# ─────────────────────────────────────────────────────────────────────────────
# Build figures for the UI and agents
# ─────────────────────────────────────────────────────────────────────────────


def build_diagnostic_figures(
    diagnostics: dict[str, Any], target_name: str = "target"
) -> list[tuple[str, go.Figure]]:
    if not diagnostics:
        return []
    y_test = diagnostics.get("y_test") or []
    y_pred = diagnostics.get("y_pred") or []
    if not y_test or not y_pred:
        return []
    if len(y_test) != len(y_pred):
        log.warning("build_diagnostic_figures | length mismatch y_test=%d y_pred=%d", len(y_test), len(y_pred))
        return []

    task = diagnostics.get("task_type", "")
    is_ts = bool(diagnostics.get("is_time_series"))
    time_index = diagnostics.get("time_index")

    figs: list[tuple[str, go.Figure]] = []

    if task == "regression":
        if is_ts and time_index:
            figs.append((
                "Forecast vs Actual over Time",
                _forecast_figure(time_index, y_test, y_pred, target_name),
            ))
        figs.append((
            "Predicted vs Actual",
            _predicted_vs_actual_figure(y_test, y_pred, target_name),
        ))
        figs.append((
            "Residuals",
            _residuals_figure(y_test, y_pred, target_name),
        ))
    elif task == "classification":
        figs.append((
            "Confusion Matrix",
            _confusion_matrix_figure(y_test, y_pred, target_name),
        ))

    return figs


def build_primary_diagnostic_figure(
    diagnostics: dict[str, Any], target_name: str = "target"
) -> go.Figure | None:
    figs = build_diagnostic_figures(diagnostics, target_name)
    return figs[0][1] if figs else None


def build_feature_importance_figure(
    model_path: str, top_n: int = 20, run_label: str = ""
) -> go.Figure | None:
    try:
        import joblib
        pipeline = joblib.load(model_path)
        estimator = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else None
        if estimator is None or not hasattr(estimator, "feature_importances_"):
            log.info("feature_importance | estimator=%s has no feature_importances_", type(estimator).__name__)
            return None

        importances = np.asarray(estimator.feature_importances_)

        names: list[str] = []
        preproc = pipeline.named_steps.get("preprocessor")
        if preproc is not None and hasattr(preproc, "get_feature_names_out"):
            try:
                names = [str(n) for n in preproc.get_feature_names_out()]
            except Exception:
                names = []
        if len(names) != len(importances):
            names = [f"feature_{i}" for i in range(len(importances))]

        order = np.argsort(importances)[::-1][:top_n]
        sel_names = [names[i] for i in order][::-1]
        sel_vals = importances[order][::-1]

        fig = go.Figure(
            data=go.Bar(
                x=sel_vals, y=sel_names, orientation="h",
                marker=dict(color="#2563eb"),
            )
        )
        title = "Feature Importance"
        if run_label:
            title = f"{title} — {run_label}"
        fig.update_layout(
            title=f"{title} (top {len(sel_names)})",
            xaxis_title="Importance",
            yaxis_title="Feature",
            template="plotly_white",
            height=max(300, 28 * len(sel_names) + 80),
        )
        return fig
    except Exception as exc:
        log.warning("build_feature_importance_figure | failed: %s", exc)
        return None


def build_leaderboard_figure(
    leaderboard: list[dict[str, Any]], task_type: str, run_label: str = ""
) -> go.Figure | None:
    if not leaderboard:
        return None
    metric_key = "f1_weighted" if task_type == "classification" else "r2"
    rows: list[tuple[str, float, str]] = []
    for entry in leaderboard:
        if entry.get("status") != "success":
            continue
        metrics = entry.get("metrics", {}) or {}
        if metric_key not in metrics:
            continue
        rows.append((str(entry.get("model", "")), float(metrics[metric_key]), entry.get("status", "")))
    if not rows:
        return None
    rows.sort(key=lambda r: r[1], reverse=True)
    fig = go.Figure(
        data=go.Bar(
            x=[r[1] for r in rows],
            y=[r[0] for r in rows],
            orientation="h",
            marker=dict(color="#7c3aed"),
            text=[f"{r[1]:.4f}" for r in rows],
            textposition="outside",
        )
    )
    title = f"Leaderboard — {metric_key}"
    if run_label:
        title = f"{title} ({run_label})"
    fig.update_layout(
        title=title,
        xaxis_title=metric_key,
        yaxis_title="Model",
        template="plotly_white",
        height=max(280, 36 * len(rows) + 80),
    )
    return fig


def build_run_comparison_figure(
    runs: list[dict[str, Any]], metric_key: str | None = None
) -> go.Figure | None:
    if not runs:
        return None
    if metric_key is None:
        task = (runs[0].get("task_type") or "").lower()
        metric_key = "f1_weighted" if task == "classification" else "r2"

    labels: list[str] = []
    values: list[float] = []
    for run in runs:
        metrics = run.get("best_metrics") or {}
        if metric_key not in metrics:
            continue
        labels.append(str(run.get("run_id", "")))
        values.append(float(metrics[metric_key]))
    if not labels:
        return None

    fig = go.Figure(
        data=go.Bar(
            x=labels, y=values,
            marker=dict(color="#16a34a"),
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Run Comparison — {metric_key}",
        xaxis_title="Run",
        yaxis_title=metric_key,
        template="plotly_white",
    )
    return fig


def build_residuals_over_time_figure(
    diagnostics: dict[str, Any], target_name: str = "target"
) -> go.Figure | None:
    if not diagnostics or not diagnostics.get("is_time_series"):
        return None
    time_index = diagnostics.get("time_index") or []
    y_test = diagnostics.get("y_test") or []
    y_pred = diagnostics.get("y_pred") or []
    if not (len(time_index) == len(y_test) == len(y_pred)) or not y_test:
        return None
    y_t = np.asarray(y_test, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    residuals = y_p - y_t

    fig = go.Figure()
    fig.add_scatter(
        x=time_index, y=residuals, mode="lines+markers",
        name="Residual",
        line=dict(color="#7c3aed"),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#dc2626")
    fig.update_layout(
        title=f"Residuals over Time — {target_name}",
        xaxis_title="Time",
        yaxis_title="Predicted − Actual",
        template="plotly_white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────────────────────────────────────


def _forecast_figure(time_index, y_test, y_pred, target_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=time_index, y=y_test, mode="lines+markers", name="Actual",
                    line=dict(color="#2563eb"))
    fig.add_scatter(x=time_index, y=y_pred, mode="lines+markers", name="Predicted",
                    line=dict(color="#dc2626", dash="dash"))
    fig.update_layout(
        title=f"Forecast vs Actual — {target_name}",
        xaxis_title="Time",
        yaxis_title=target_name,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _predicted_vs_actual_figure(y_test, y_pred, target_name: str) -> go.Figure:
    y_t = np.asarray(y_test, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    mn = float(min(y_t.min(), y_p.min()))
    mx = float(max(y_t.max(), y_p.max()))

    fig = go.Figure()
    fig.add_scatter(x=y_t, y=y_p, mode="markers",
                    name="Test predictions", opacity=0.55,
                    marker=dict(color="#2563eb", size=6))
    fig.add_scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                    name="Perfect prediction",
                    line=dict(color="#16a34a", dash="dash"))
    fig.update_layout(
        title=f"Predicted vs Actual — {target_name}",
        xaxis_title=f"Actual {target_name}",
        yaxis_title=f"Predicted {target_name}",
        template="plotly_white",
    )
    return fig


def _residuals_figure(y_test, y_pred, target_name: str) -> go.Figure:
    y_t = np.asarray(y_test, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    residuals = y_p - y_t

    fig = go.Figure()
    fig.add_scatter(x=y_t, y=residuals, mode="markers",
                    name="Residuals", opacity=0.55,
                    marker=dict(color="#7c3aed", size=6))
    fig.add_hline(y=0, line_dash="dash", line_color="#dc2626")
    fig.update_layout(
        title=f"Residuals — {target_name}",
        xaxis_title=f"Actual {target_name}",
        yaxis_title="Residual (predicted − actual)",
        template="plotly_white",
    )
    return fig


def _confusion_matrix_figure(y_test, y_pred, target_name: str) -> go.Figure:
    y_t = pd.Series(y_test)
    y_p = pd.Series(y_pred)
    labels = sorted(set(y_t.unique()) | set(y_p.unique()), key=lambda v: str(v))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    for actual, pred in zip(y_t, y_p):
        if actual in label_to_idx and pred in label_to_idx:
            matrix[label_to_idx[actual]][label_to_idx[pred]] += 1

    label_strs = [str(label) for label in labels]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=label_strs,
            y=label_strs,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Confusion Matrix — {target_name}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _time_axis_from_series(
    series: pd.Series,
) -> tuple[list[str] | None, pd.Series | None]:
    s = series.reset_index(drop=True)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y-%m-%d").tolist(), s.astype("int64")
    try:
        parsed = pd.to_datetime(s, errors="raise")
        return parsed.dt.strftime("%Y-%m-%d").tolist(), parsed.astype("int64")
    except Exception:
        labels = [str(v) for v in s]
        sort_key = pd.Series(list(range(len(s))))
        try:
            sort_key = s.rank(method="first").astype("int64")
        except Exception:
            pass
        return labels, sort_key


def _detect_time_axis(
    df: pd.DataFrame,
) -> tuple[list[str] | None, pd.Series | None]:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            series = df[col]
            return (
                series.dt.strftime("%Y-%m-%d").tolist(),
                series.astype("int64").reset_index(drop=True),
            )
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="raise")
                return (
                    parsed.dt.strftime("%Y-%m-%d").tolist(),
                    parsed.astype("int64").reset_index(drop=True),
                )
            except Exception:
                continue
    if "year" in df.columns and "month" in df.columns:
        try:
            year = df["year"].astype(int).reset_index(drop=True)
            month = df["month"].astype(int).reset_index(drop=True)
            sort_key = year * 12 + month
            day = (
                df["day"].astype(int).reset_index(drop=True)
                if "day" in df.columns else pd.Series([1] * len(df))
            )
            if "day" in df.columns:
                sort_key = sort_key * 31 + day
            labels = [
                f"{y:04d}-{m:02d}-{d:02d}" if "day" in df.columns else f"{y:04d}-{m:02d}"
                for y, m, d in zip(year, month, day)
            ]
            return labels, sort_key
        except Exception:
            pass
    return None, None


def _to_jsonable_list(series: pd.Series) -> list:
    if pd.api.types.is_numeric_dtype(series):
        return [None if pd.isna(v) else float(v) for v in series]
    return [None if pd.isna(v) else str(v) for v in series]
