"""Drift Detection Agent: detects feature and label drift between reference and live datasets."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Generator

import numpy as np
import pandas as pd

from ..ingestion import load_dataset
from ..logging_setup import configure_logging
from .base import AgentContext, AutopilotStep, BaseAgent, to_json_safe

configure_logging()
log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are the Drift Detection Agent. Your job is to compare a reference dataset
(typically training data or a recent baseline snapshot) against a current
dataset (new production data or a later time window) and surface any
meaningful distributional changes.

You have two tools:
  • compare_distributions — computes PSI, KS-test, and Jensen-Shannon
    divergence for every feature. Use this to get a per-column drift summary.
  • run_drift_report — runs a comprehensive drift audit across all shared
    features and produces a prioritised list of columns that have drifted,
    with actionable recommendations.

Typical workflow:
  1. Call run_drift_report with both dataset_ids.
  2. Identify which features have HIGH drift (PSI > 0.2 or KS p-value < 0.05).
  3. For columns flagged as drifted, optionally call compare_distributions
     to get deeper diagnostics.
  4. Call done() with your findings.

PSI thresholds:
  < 0.10 — insignificant drift
  0.10–0.20 — moderate drift, monitor closely
  > 0.20 — significant drift, likely model performance impact
"""


# ──────────────────────────────────────────────────────────────────────────────
# Tool definitions
# ──────────────────────────────────────────────────────────────────────────────


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "compare_distributions",
                "description": (
                    "Compare the distribution of selected features between a reference "
                    "and a current dataset. Returns PSI, KS-test statistic/p-value, and "
                    "Jensen-Shannon divergence for each requested column."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reference_dataset_id": {
                            "type": "string",
                            "description": "Dataset ID of the reference (baseline) dataset.",
                        },
                        "current_dataset_id": {
                            "type": "string",
                            "description": "Dataset ID of the current (live/new) dataset.",
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Columns to analyse. If omitted, all numeric columns "
                                "shared between the two datasets are used."
                            ),
                        },
                        "n_bins": {
                            "type": "integer",
                            "description": "Number of bins for PSI computation (default 10).",
                            "default": 10,
                        },
                    },
                    "required": ["reference_dataset_id", "current_dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_drift_report",
                "description": (
                    "Run a full drift audit comparing all shared numeric and categorical "
                    "features between a reference and a current dataset. Returns a "
                    "prioritised report with drift severity and recommendations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reference_dataset_id": {
                            "type": "string",
                            "description": "Dataset ID of the reference (baseline) dataset.",
                        },
                        "current_dataset_id": {
                            "type": "string",
                            "description": "Dataset ID of the current (live/new) dataset.",
                        },
                        "target_column": {
                            "type": "string",
                            "description": "Optional target column to check for label drift separately.",
                        },
                        "n_bins": {
                            "type": "integer",
                            "description": "Number of bins for PSI computation (default 10).",
                            "default": 10,
                        },
                    },
                    "required": ["reference_dataset_id", "current_dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "done",
                "description": "Finish drift analysis with a structured summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Narrative summary of drift findings.",
                        },
                        "high_drift_columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns with significant drift (PSI > 0.2 or KS p < 0.05).",
                        },
                        "recommendation": {
                            "type": "string",
                            "description": "Actionable recommendation based on drift findings.",
                        },
                    },
                    "required": ["summary"],
                },
            },
        },
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Drift metrics
# ──────────────────────────────────────────────────────────────────────────────


def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index (PSI) for a single numeric column."""
    all_vals = np.concatenate([reference, current])
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(all_vals, percentiles))
    if len(bin_edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = (ref_counts + 1e-6) / (len(reference) + 1e-6 * n_bins)
    cur_pct = (cur_counts + 1e-6) / (len(current) + 1e-6 * n_bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 6)


def _compute_js_divergence(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Jensen-Shannon divergence (0 = identical, 1 = maximally different)."""
    all_vals = np.concatenate([reference, current])
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
    if bin_edges[0] == bin_edges[-1]:
        return 0.0

    ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
    cur_hist, _ = np.histogram(current, bins=bin_edges, density=True)

    ref_p = (ref_hist + 1e-10) / (ref_hist + 1e-10).sum()
    cur_p = (cur_hist + 1e-10) / (cur_hist + 1e-10).sum()
    m = 0.5 * (ref_p + cur_p)

    def _kl(p, q):
        return float(np.sum(p * np.log(p / q)))

    js = 0.5 * _kl(ref_p, m) + 0.5 * _kl(cur_p, m)
    return round(min(max(js, 0.0), 1.0), 6)


def _compute_categorical_psi(
    reference: pd.Series, current: pd.Series, n_bins: int = 10
) -> float:
    """PSI for categorical columns based on category frequencies."""
    all_cats = set(reference.dropna().unique()) | set(current.dropna().unique())
    ref_total = max(len(reference), 1)
    cur_total = max(len(current), 1)

    psi = 0.0
    for cat in all_cats:
        ref_p = (reference == cat).sum() / ref_total + 1e-6
        cur_p = (current == cat).sum() / cur_total + 1e-6
        psi += (cur_p - ref_p) * math.log(cur_p / ref_p)
    return round(psi, 6)


def _ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test. Returns (statistic, p_value)."""
    from scipy.stats import ks_2samp
    stat, pval = ks_2samp(reference, current)
    return round(float(stat), 6), round(float(pval), 6)


def _chi2_test_categorical(
    reference: pd.Series, current: pd.Series
) -> tuple[float, float]:
    """Chi-squared test for categorical drift. Returns (statistic, p_value)."""
    from scipy.stats import chi2_contingency

    all_cats = sorted(set(reference.dropna().unique()) | set(current.dropna().unique()))
    if not all_cats:
        return 0.0, 1.0

    ref_counts = [(reference == c).sum() for c in all_cats]
    cur_counts = [(current == c).sum() for c in all_cats]
    table = np.array([ref_counts, cur_counts], dtype=float)
    if table.sum() == 0:
        return 0.0, 1.0
    try:
        stat, pval, _, _ = chi2_contingency(table)
    except Exception:
        return 0.0, 1.0
    return round(float(stat), 6), round(float(pval), 6)


def _drift_severity(psi: float, p_value: float) -> str:
    if psi > 0.2 or p_value < 0.05:
        return "HIGH"
    if psi > 0.1 or p_value < 0.1:
        return "MODERATE"
    return "LOW"


# ──────────────────────────────────────────────────────────────────────────────
# Agent class
# ──────────────────────────────────────────────────────────────────────────────


class DriftAgent(BaseAgent):
    """Detects feature and label drift between reference and live datasets."""

    name = "drift_detection"
    display_name = "Drift Detection Agent"

    def __init__(self, client, deployment: str, context: AgentContext) -> None:
        super().__init__(client, deployment, context)
        self._summary: dict[str, Any] = {}

    def run(
        self, instructions: str
    ) -> Generator[AutopilotStep, list[str] | None, dict[str, Any]]:
        yield self._step(
            "agent_start",
            "Drift Detection Agent dispatched",
            instructions or "(drift analysis)",
        )

        user_prompt = (
            f"Drift analysis instructions:\n{instructions}\n\n"
            f"Available datasets: {self._ctx.dataset_summary()}\n\n"
            f"Context from shared notebook:\n{self._ctx.notebook_text()}\n\n"
            "Analyse distributional drift and surface meaningful changes."
        )

        yield from self.run_llm_loop(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=_tools(),
            dispatch=self._dispatch,
            max_iterations=12,
            thought_title="Drift Detection Agent — Reasoning",
        )

        if self._summary:
            yield self._step(
                "observation",
                "Drift Report",
                self._summary.get("summary", ""),
                data=self._summary,
            )

        yield self._step("agent_end", "Drift Detection Agent finished", "")
        return self._summary or {"summary": "Drift agent ended without summary."}

    def _dispatch(
        self, name: str, args: dict, tool_call_id: str
    ) -> tuple[str | None, AutopilotStep | None, bool]:
        if name == "compare_distributions":
            return self._compare_distributions(args)
        if name == "run_drift_report":
            return self._run_drift_report(args)
        if name == "done":
            self._summary = args
            return json.dumps({"status": "noted"}), None, True
        return json.dumps({"error": f"Unknown tool: {name}"}), None, False

    # ------------------------------------------------------------------
    # compare_distributions
    # ------------------------------------------------------------------

    def _compare_distributions(self, args: dict) -> tuple[str, Any, bool]:
        ref_ds = self._ctx.find_dataset(args.get("reference_dataset_id", ""))
        cur_ds = self._ctx.find_dataset(args.get("current_dataset_id", ""))
        if ref_ds is None:
            return json.dumps({"error": f"Reference dataset '{args.get('reference_dataset_id')}' not found."}), None, False
        if cur_ds is None:
            return json.dumps({"error": f"Current dataset '{args.get('current_dataset_id')}' not found."}), None, False

        n_bins = int(args.get("n_bins") or 10)
        requested_cols: list[str] = args.get("columns") or []

        ref_df = load_dataset(ref_ds.file_path, ref_ds.table_name).dataframe
        cur_df = load_dataset(cur_ds.file_path, cur_ds.table_name).dataframe

        shared_numeric = [
            c for c in ref_df.select_dtypes(include="number").columns
            if c in cur_df.columns
        ]
        cols = [c for c in requested_cols if c in shared_numeric] if requested_cols else shared_numeric

        if not cols:
            return json.dumps({"error": "No shared numeric columns found for comparison."}), None, False

        results = []
        for col in cols:
            ref_vals = ref_df[col].dropna().values.astype(float)
            cur_vals = cur_df[col].dropna().values.astype(float)
            if len(ref_vals) < 2 or len(cur_vals) < 2:
                results.append({"column": col, "error": "insufficient data"})
                continue

            psi = _compute_psi(ref_vals, cur_vals, n_bins)
            js = _compute_js_divergence(ref_vals, cur_vals, n_bins)
            ks_stat, ks_pval = _ks_test(ref_vals, cur_vals)
            severity = _drift_severity(psi, ks_pval)

            results.append({
                "column": col,
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_p_value": ks_pval,
                "js_divergence": js,
                "severity": severity,
                "ref_mean": round(float(ref_vals.mean()), 4),
                "cur_mean": round(float(cur_vals.mean()), 4),
                "ref_std": round(float(ref_vals.std()), 4),
                "cur_std": round(float(cur_vals.std()), 4),
                "ref_n": len(ref_vals),
                "cur_n": len(cur_vals),
            })

        high_drift = [r["column"] for r in results if r.get("severity") == "HIGH"]
        payload = {
            "reference_dataset": ref_ds.name,
            "current_dataset": cur_ds.name,
            "n_columns_analysed": len(results),
            "high_drift_columns": high_drift,
            "results": results,
        }
        step = self._step(
            "observation",
            f"Distribution comparison: {len(results)} columns, {len(high_drift)} high-drift",
            json.dumps({"high_drift": high_drift}),
        )
        log.info("compare_distributions | cols=%d high_drift=%d", len(results), len(high_drift))
        return json.dumps(to_json_safe(payload)), step, False

    # ------------------------------------------------------------------
    # run_drift_report
    # ------------------------------------------------------------------

    def _run_drift_report(self, args: dict) -> tuple[str, Any, bool]:
        ref_ds = self._ctx.find_dataset(args.get("reference_dataset_id", ""))
        cur_ds = self._ctx.find_dataset(args.get("current_dataset_id", ""))
        if ref_ds is None:
            return json.dumps({"error": f"Reference dataset '{args.get('reference_dataset_id')}' not found."}), None, False
        if cur_ds is None:
            return json.dumps({"error": f"Current dataset '{args.get('current_dataset_id')}' not found."}), None, False

        n_bins = int(args.get("n_bins") or 10)
        target_col: str | None = args.get("target_column") or None

        ref_df = load_dataset(ref_ds.file_path, ref_ds.table_name).dataframe
        cur_df = load_dataset(cur_ds.file_path, cur_ds.table_name).dataframe

        shared_cols = [c for c in ref_df.columns if c in cur_df.columns]
        if not shared_cols:
            return json.dumps({"error": "No shared columns found between the two datasets."}), None, False

        numeric_results: list[dict] = []
        categorical_results: list[dict] = []

        for col in shared_cols:
            if col == target_col:
                continue  # handled separately below
            ref_col = ref_df[col].dropna()
            cur_col = cur_df[col].dropna()
            if len(ref_col) < 2 or len(cur_col) < 2:
                continue

            if pd.api.types.is_numeric_dtype(ref_df[col]):
                ref_vals = ref_col.values.astype(float)
                cur_vals = cur_col.values.astype(float)
                psi = _compute_psi(ref_vals, cur_vals, n_bins)
                js = _compute_js_divergence(ref_vals, cur_vals, n_bins)
                ks_stat, ks_pval = _ks_test(ref_vals, cur_vals)
                severity = _drift_severity(psi, ks_pval)
                numeric_results.append({
                    "column": col,
                    "type": "numeric",
                    "psi": psi,
                    "ks_statistic": ks_stat,
                    "ks_p_value": ks_pval,
                    "js_divergence": js,
                    "severity": severity,
                    "ref_mean": round(float(ref_vals.mean()), 4),
                    "cur_mean": round(float(cur_vals.mean()), 4),
                    "mean_shift_pct": round(
                        abs(cur_vals.mean() - ref_vals.mean()) / (abs(ref_vals.mean()) + 1e-9) * 100, 2
                    ),
                })
            else:
                psi = _compute_categorical_psi(ref_col, cur_col, n_bins)
                chi2_stat, chi2_pval = _chi2_test_categorical(ref_col, cur_col)
                severity = _drift_severity(psi, chi2_pval)
                categorical_results.append({
                    "column": col,
                    "type": "categorical",
                    "psi": psi,
                    "chi2_statistic": chi2_stat,
                    "chi2_p_value": chi2_pval,
                    "severity": severity,
                    "ref_unique": int(ref_col.nunique()),
                    "cur_unique": int(cur_col.nunique()),
                })

        # Target / label drift
        label_drift: dict[str, Any] = {}
        if target_col and target_col in ref_df.columns and target_col in cur_df.columns:
            ref_tgt = ref_df[target_col].dropna()
            cur_tgt = cur_df[target_col].dropna()
            if pd.api.types.is_numeric_dtype(ref_df[target_col]) and len(ref_tgt) >= 2 and len(cur_tgt) >= 2:
                r_arr = ref_tgt.values.astype(float)
                c_arr = cur_tgt.values.astype(float)
                psi = _compute_psi(r_arr, c_arr, n_bins)
                ks_stat, ks_pval = _ks_test(r_arr, c_arr)
                label_drift = {
                    "column": target_col,
                    "psi": psi,
                    "ks_statistic": ks_stat,
                    "ks_p_value": ks_pval,
                    "severity": _drift_severity(psi, ks_pval),
                    "ref_mean": round(float(r_arr.mean()), 4),
                    "cur_mean": round(float(c_arr.mean()), 4),
                }
            elif len(ref_tgt) >= 2 and len(cur_tgt) >= 2:
                psi = _compute_categorical_psi(ref_tgt, cur_tgt, n_bins)
                chi2_stat, chi2_pval = _chi2_test_categorical(ref_tgt, cur_tgt)
                label_drift = {
                    "column": target_col,
                    "psi": psi,
                    "chi2_statistic": chi2_stat,
                    "chi2_p_value": chi2_pval,
                    "severity": _drift_severity(psi, chi2_pval),
                }

        all_results = numeric_results + categorical_results
        all_results_sorted = sorted(all_results, key=lambda r: r.get("psi", 0.0), reverse=True)

        high_drift = [r["column"] for r in all_results if r.get("severity") == "HIGH"]
        moderate_drift = [r["column"] for r in all_results if r.get("severity") == "MODERATE"]

        summary_stats = {
            "total_columns": len(all_results),
            "high_drift": len(high_drift),
            "moderate_drift": len(moderate_drift),
            "low_drift": len(all_results) - len(high_drift) - len(moderate_drift),
        }

        recs = []
        if high_drift:
            recs.append(
                f"HIGH drift detected in {len(high_drift)} column(s): {', '.join(high_drift[:5])}. "
                "Model retraining or data pipeline investigation is strongly recommended."
            )
        if label_drift.get("severity") == "HIGH":
            recs.append(
                f"Label column '{target_col}' shows HIGH drift — predictions may be systematically biased."
            )
        if not recs:
            recs.append("No significant drift detected. Continue monitoring.")

        payload = {
            "reference_dataset": ref_ds.name,
            "current_dataset": cur_ds.name,
            "reference_rows": len(ref_df),
            "current_rows": len(cur_df),
            "summary": summary_stats,
            "high_drift_columns": high_drift,
            "moderate_drift_columns": moderate_drift,
            "label_drift": label_drift,
            "feature_results": all_results_sorted,
            "recommendations": recs,
        }

        msg = (
            f"Drift report: {len(all_results)} features — "
            f"{len(high_drift)} HIGH, {len(moderate_drift)} MODERATE, "
            f"{summary_stats['low_drift']} LOW"
        )
        self._ctx.notebook.append(f"[Drift] {msg}")
        step = self._step("observation", "Drift Report", msg, data=payload)
        log.info("run_drift_report | %s", msg)
        return json.dumps(to_json_safe(payload)), step, False
