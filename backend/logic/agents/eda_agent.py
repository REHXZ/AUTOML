"""EDA Agent: profiles datasets and generates charts with vision feedback."""

from __future__ import annotations

import json
import logging
from typing import Any, Generator

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..dtype_coercion import coerce_dtypes
from ..ingestion import load_dataset
from backend.server.logging_setup import configure_logging
from ..profiling import profile_dataframe
from .base import (
    AgentContext,
    AutopilotStep,
    BaseAgent,
    to_json_safe,
    vision_tool_content,
)

configure_logging()
log = logging.getLogger(__name__)

# Chart types that require at least one column parameter in params.
_PARAM_REQUIRED_CHARTS = frozenset({
    "histogram", "bar", "scatter", "box", "violin", "pairplot", "line", "qq_plot",
})
# All known per-chart param keys — used to detect flat args from LLM.
_KNOWN_PARAM_KEYS = frozenset({
    "column", "bins", "top_n", "x_column", "y_column",
    "color_column", "group_by", "columns",
    # new chart / analysis params
    "target_column", "time_column", "period", "model", "nlags", "normalize",
})


_SYSTEM_PROMPT = """\
# Role & Objective
You are the EDA Agent — a senior data scientist with 10+ years specialising
in exploratory analysis, anomaly detection, and translating statistical
findings into actionable modelling decisions.

════════════════════════════════════════════════════════════════════════
# ANALYSIS PROTOCOL — FOLLOW THIS ORDER

## Step 1 — Orient (ALWAYS first)
Call profile_dataset on every available dataset.
Reason aloud: shape, dtypes, missing % per column, basic statistics.
Flag immediately: duplicate column names, all-null columns, columns
that look like IDs (monotonically increasing ints, UUID-like strings).

## Step 2 — Target Deep-Dive
  For CLASSIFICATION targets:
    run_analysis class_balance → flag imbalance ratio.
    Imbalance > 3:1 → recommend SMOTE.
    Imbalance > 10:1 → recommend SMOTE + class_weight="balanced".
    create_chart bar on target to visualise class counts.

  For REGRESSION targets:
    run_analysis normality_test on target column.
    Skewness > 1.5 → recommend target_log_transform IN PLACE.
    create_chart histogram on target + create_chart qq_plot for normality.
    run_analysis target_correlation → rank features by linear signal.
    run_analysis mutual_information → rank features by non-linear signal.

## Step 3 — Feature Quality Audit
  For every numeric column:
    - Missing > 50 % → recommend drop_high_missing.
    - Missing 5-50 % → recommend knn_impute + add_missing_indicators.
    - Missing < 5 % → recommend impute_missing (median).
    - Skewness > 1.5 → recommend log_transform (adds new col).
    - Near-zero variance → recommend drop_constant.

  For every categorical column:
    - Cardinality > 50 → recommend target_encode or hash_encode.
    - Cardinality 5-50 → recommend ordinal_encode or one_hot_encode.
    - Cardinality < 5 → recommend one_hot_encode.
    - Check for date-like strings → recommend datetime_parse + encode_dates.

  run_analysis vif on all numeric features.
    VIF > 10 → name the collinear pair; recommend drop_correlated.
  run_analysis outlier_summary.
    Outlier rate > 10 % in a column → recommend winsorize.

## Step 4 — LEAKAGE DETECTION (CRITICAL — NEVER SKIP)
  Leakage is the #1 cause of model results that look too good.
  Check ALL of the following:

  ① TARGET PROXY COLUMNS: any feature correlated > 0.95 with target.
    run_analysis target_correlation. If r > 0.95 → name it, flag as
    "PROBABLE LEAKAGE — drop before modeling".

  ② SEMANTIC LEAKAGE: scan column names for target-derived words:
    "result", "flag", "label", "outcome", "y_", "target", "score",
    "approved", "churn", "default". Name every suspect.

  ③ ID LEAKAGE: ID-like columns (row_id, order_id, customer_id).
    create_chart scatter (id vs target). A trend = index leakage.
    Recommend drop_columns for pure IDs.

  ④ FUTURE LEAKAGE: any column describing what happened AFTER the
    prediction window. Look for column names with "final", "outcome",
    "actual", "post", "result". These must be dropped.

  ⑤ AGGREGATION LEAKAGE: target statistics computed on the full
    dataset before splitting (e.g., customer mean order value that
    includes the test period). Flag "requires careful target encoding
    with out-of-fold computation only".

## Step 5 — Time-Series Checks (ONLY for temporal datasets)
  If ANY column is datetime-like or appears to be an ordered index:
  ① run_analysis seasonal_decompose.
    Seasonal pattern present → recommend fourier_features or cyclical_encode.
    Trend present → recommend detrending or differencing before ARIMA.
  ② run_analysis stationarity_test (ADF + KPSS).
    Both tests reject stationarity → recommend differencing.
    ADF rejects but KPSS does not → trend-stationary → demean or detrend.
  ③ run_analysis acf_pacf.
    Significant ACF at lags [k] → recommend create_lag_features lags=[k, 2k].
    Rapid ACF decay + spike in PACF at p → AR(p) structure; lag p is key.
    ACF decays slowly (long memory) → recommend longer rolling windows.
  Create a chart: create_chart line of target over time to visualise trend.

## Step 6 — Relationship Exploration
  create_chart correlation_heatmap (all numeric features).
  create_chart scatter for top-3 MI features vs target.
  create_chart pairplot for up to 4 highest-MI numeric features.
  For classification: create_chart box (target as group) for each
    numeric feature — look for class separation vs overlap.
  For regression: look for non-linear patterns in scatter plots that
    suggest interaction_features or polynomial_features.

## Step 7 — Statistical Tests (choose based on data)
  Selection guide:
  - 2 groups, normal data:           ttest
  - 2 groups, non-normal / ordinal:  mannwhitney
  - 3+ groups, normal data:          anova
  - 3+ groups, non-normal:           kruskal_wallis
  - 2 categorical columns:           chi2_test
  - All pairwise numeric with p-val: correlation_significance
  Report: statistic, p-value, and a plain-English interpretation.

════════════════════════════════════════════════════════════════════════
# INTERPRETATION HEURISTICS

| Finding                          | Meaning                        | Recommendation                        |
|----------------------------------|--------------------------------|---------------------------------------|
| Skewness > 1.5 on target         | Long right tail, inflated RMSE | target_log_transform in-place         |
| Skewness > 1.5 on feature        | Outlier-dominated signal       | log_transform (adds new col)          |
| Missing > 50 %                   | Column unreliable              | drop_high_missing                     |
| Missing 5-50 %                   | Potential informative missing  | knn_impute + add_missing_indicators   |
| VIF > 10                         | Multicollinearity              | drop_correlated or PCA                |
| Feature corr with target > 0.95  | PROBABLE LEAKAGE               | drop_columns, flag in findings        |
| Class imbalance > 5:1            | Accuracy metric is misleading  | SMOTE + class_weight="balanced"       |
| ADF p > 0.05                     | Non-stationary series          | Differencing or ARIMA with d=1        |
| ACF significant at lag k         | Seasonal / AR pattern          | create_lag_features lags=[k, 2k, 3k] |
| Outlier rate > 10 %              | Noisy feature                  | winsorize or zscore_outlier_removal   |
| Cardinality > 50                 | High-cardinality categorical   | target_encode or hash_encode          |
| Near-zero variance               | No predictive power            | drop_constant                         |

════════════════════════════════════════════════════════════════════════
# VISION INSTRUCTIONS
After every create_chart call you receive the chart image. For each:
  1. Describe exactly what you SEE (shapes, outliers, trends, gaps, clusters).
  2. State what it MEANS for the modelling strategy.
  3. Call record_finding with both the observation and the implication.
Never say "I cannot see the chart" — you can. Be specific and descriptive.

════════════════════════════════════════════════════════════════════════
# COMPLETION CRITERIA
You are done when you can answer ALL of the following:
  ① Task type and recommended primary metric?
  ② Top 5-10 features by signal (MI or correlation)?
  ③ Any leakage suspects? Name them explicitly.
  ④ Data quality issues FE must fix (missing, outliers, encoding)?
  ⑤ Class imbalance or skewed target? What resampling is needed?
  ⑥ For time-series: seasonality present? Suggested lag order?

Call record_finding for each major insight.
Call done(summary) with these JSON keys:
  candidate_targets, leakage_suspects, recommended_transforms,
  class_imbalance_ratio, time_series_insights, top_features, next_steps.
"""


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "profile_dataset",
                "description": "Return row count, column roles, missing %, duplicates, and numeric stats.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_chart",
                "description": (
                    "Generate a chart and receive the rendered image for visual analysis. "
                    "chart_type: "
                    "'histogram' (params: column, bins?); "
                    "'bar' (params: column, top_n?); "
                    "'scatter' (params: x_column, y_column, color_column?); "
                    "'correlation_heatmap'; "
                    "'box' (params: column, group_by?); "
                    "'missing_heatmap'; "
                    "'violin' (params: column, group_by?); "
                    "'pairplot' (params: columns, color_column?) — up to 4 numeric columns; "
                    "'line' (params: x_column, y_column or columns, color_column?) — time-series or ordered line chart; "
                    "'qq_plot' (params: column) — normal Q-Q plot for distribution check."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "chart_type": {
                            "type": "string",
                            "enum": [
                                "histogram", "bar", "scatter",
                                "correlation_heatmap", "box", "missing_heatmap",
                                "violin", "pairplot", "line", "qq_plot",
                            ],
                        },
                        "params": {"type": "object"},
                    },
                    "required": ["dataset_id", "chart_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_analysis",
                "description": (
                    "Run a statistical or time-series analysis and receive structured results "
                    "(plus an optional chart image). "
                    "analysis_type: "
                    "'class_balance' (params: column) — class counts and imbalance ratio; "
                    "'target_correlation' (params: target_column) — feature correlations with the target; "
                    "'mutual_information' (params: target_column) — feature MI scores; "
                    "'normality_test' (params: column) — Shapiro-Wilk / D'Agostino skew+kurtosis; "
                    "'vif' — Variance Inflation Factor for multicollinearity; "
                    "'outlier_summary' — IQR and z-score outlier counts per numeric column; "
                    "'seasonal_decompose' (params: column, time_column, period?, model?) — "
                        "trend/seasonal/residual decomposition (requires statsmodels); "
                    "'stationarity_test' (params: column) — ADF + KPSS stationarity tests; "
                    "'acf_pacf' (params: column, nlags?) — ACF and PACF with chart."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "analysis_type": {
                            "type": "string",
                            "enum": [
                                "class_balance", "target_correlation",
                                "mutual_information", "normality_test",
                                "vif", "outlier_summary",
                                "seasonal_decompose", "stationarity_test",
                                "acf_pacf",
                            ],
                        },
                        "params": {"type": "object"},
                    },
                    "required": ["dataset_id", "analysis_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_finding",
                "description": "Write a single observation bullet to the shared notebook.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "done",
                "description": "Report finished. Provide a structured JSON summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "candidate_targets": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "problematic_columns": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "recommended_transformations": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "suspected_leakage": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "next_moves": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "narrative": {"type": "string"},
                    },
                    "required": ["narrative"],
                },
            },
        },
    ]


class EdaAgent(BaseAgent):
    """Profiles datasets and creates EDA charts on the Scientist's behalf."""

    name = "eda"
    display_name = "EDA Agent"

    def __init__(self, client, deployment: str, context: AgentContext) -> None:
        super().__init__(client, deployment, context)
        self._summary: dict[str, Any] = {}

    def run(
        self, instructions: str
    ) -> Generator[AutopilotStep, list[str] | None, dict[str, Any]]:
        yield self._step(
            "agent_start",
            "EDA Agent dispatched",
            instructions or "(no specific instructions — explore broadly)",
        )

        datasets = self._ctx.list_datasets()
        dataset_index = "\n".join(
            f"- id={d.id} name={d.name} rows={d.row_count} cols={d.column_count}"
            for d in datasets
        ) or "(no datasets registered)"

        user_prompt = (
            f"Scientist's instructions:\n{instructions}\n\n"
            f"Available datasets:\n{dataset_index}\n\n"
            f"Project user-stated goal: {self._ctx.user_goal or '(none)'}\n\n"
            "Begin exploration."
        )

        yield from self.run_llm_loop(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=_tools(),
            dispatch=self._dispatch,
            max_iterations=30,
            thought_title="EDA Agent — Reasoning",
        )

        yield self._step("agent_end", "EDA Agent finished", "")
        return self._summary or {"narrative": "EDA agent ended without summary."}

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self, name: str, args: dict, tool_call_id: str
    ) -> tuple[str | list | None, AutopilotStep | None, bool]:
        if name == "profile_dataset":
            return self._profile(args.get("dataset_id", ""))
        if name == "create_chart":
            return self._chart(args)
        if name == "run_analysis":
            return self._analysis(args)
        if name == "record_finding":
            return self._record(args.get("text", ""))
        if name == "done":
            self._summary = to_json_safe(args)
            return json.dumps({"status": "noted"}), None, True
        return json.dumps({"error": f"Unknown tool: {name}"}), None, False

    def _load_coerced(self, ds):
        """Load a dataset and apply automatic dtype coercion. Returns (df, changes)."""
        loaded = load_dataset(ds.file_path, ds.table_name)
        df, changes = coerce_dtypes(loaded.dataframe)
        if changes:
            log.info(
                "dtype_coercion | dataset=%s coerced %d column(s): %s",
                ds.name, len(changes), changes,
            )
        return df, changes

    def _profile(self, dataset_id: str) -> tuple[str, AutopilotStep | None, bool]:
        ds = self._ctx.find_dataset(dataset_id)
        if ds is None:
            return json.dumps({"error": f"Dataset '{dataset_id}' not found."}), None, False
        df, coercions = self._load_coerced(ds)
        profile = profile_dataframe(df)
        trimmed = {
            **profile,
            "columns": [
                {**col, "sample_values": col.get("sample_values", [])[:3]}
                for col in profile.get("columns", [])
            ],
        }
        if coercions:
            trimmed["dtype_coercions"] = coercions
        step = self._step(
            "tool_result",
            f"Profiled: {ds.name}",
            (
                f"{profile['row_count']} rows × {profile['column_count']} cols | "
                f"{profile['missing_pct']:.1f}% missing | "
                f"{profile['duplicate_rows']} duplicates"
                + (f" | {len(coercions)} dtype(s) coerced" if coercions else "")
            ),
            data={"dataset_name": ds.name, "profile": to_json_safe(trimmed)},
        )
        return json.dumps(to_json_safe(trimmed)), step, False

    def _chart(self, args: dict) -> tuple[str | list, AutopilotStep | None, bool]:
        dataset_id = args.get("dataset_id", "")
        chart_type = args.get("chart_type", "")
        params: dict = args.get("params") or {}

        # ── Flat-params fallback ──────────────────────────────────────────────
        # The LLM sometimes sends chart-specific keys at the top level of the
        # tool call instead of nested under "params".  Detect and promote them
        # so per-column charts still work, and emit a warning so we can track it.
        if not params:
            flat = {k: v for k, v in args.items() if k in _KNOWN_PARAM_KEYS}
            if flat:
                log.warning(
                    "create_chart | flat-params detected for chart_type=%r — "
                    "LLM sent %s at top level instead of nested under 'params'; promoting.",
                    chart_type, list(flat),
                )
                params = flat

        if not params and chart_type in _PARAM_REQUIRED_CHARTS:
            log.warning(
                "create_chart | chart_type=%r requires params but received none — "
                "full args: %s",
                chart_type, args,
            )

        ds = self._ctx.find_dataset(dataset_id)
        if ds is None:
            log.warning("create_chart | dataset_id=%r not found", dataset_id)
            return (
                json.dumps({"error": f"Dataset '{dataset_id}' not found."}),
                None,
                False,
            )

        log.info(
            "create_chart | dataset=%s chart_type=%r params=%s",
            ds.name, chart_type, params,
        )

        df, _ = self._load_coerced(ds)
        fig, title, description = _build_figure(
            df,
            ds.name,
            chart_type,
            params,
        )
        if fig is None:
            log.warning(
                "create_chart | FAILED dataset=%s chart_type=%r error=%r",
                ds.name, chart_type, description,
            )
            return json.dumps({"error": description}), None, False

        log.info("create_chart | OK dataset=%s chart_type=%r title=%r", ds.name, chart_type, title)
        step = self._step(
            "chart",
            title,
            description,
            data={
                "figure": fig,
                "dataset_name": ds.name,
                "dataset_id": ds.id,
                "chart_type": chart_type,
                "chart_params": dict(params),
            },
        )
        result_text = json.dumps(
            {"chart": title, "description": description, "dataset": ds.name}
        )
        return vision_tool_content(result_text, fig), step, False

    def _record(self, text: str) -> tuple[str, AutopilotStep, bool]:
        text = (text or "").strip()
        if text:
            self._ctx.notebook.append(f"[EDA] {text}")
        return (
            json.dumps({"recorded": True, "notebook_size": len(self._ctx.notebook)}),
            self._step("observation", "EDA finding", text),
            False,
        )

    def _analysis(self, args: dict) -> tuple[str | list, AutopilotStep | None, bool]:
        dataset_id = args.get("dataset_id", "")
        analysis_type = args.get("analysis_type", "")
        params: dict = args.get("params") or {}

        ds = self._ctx.find_dataset(dataset_id)
        if ds is None:
            return json.dumps({"error": f"Dataset '{dataset_id}' not found."}), None, False

        loaded = load_dataset(ds.file_path, ds.table_name)
        log.info("run_analysis | dataset=%s analysis_type=%s params=%s", ds.name, analysis_type, params)

        result, fig, title = _run_analysis(loaded.dataframe, ds.name, analysis_type, params)
        if "error" in result:
            log.warning("run_analysis | FAILED dataset=%s type=%s error=%s", ds.name, analysis_type, result["error"])
            return json.dumps(result), None, False

        log.info("run_analysis | OK dataset=%s type=%s title=%s", ds.name, analysis_type, title)
        step = self._step(
            "chart" if fig is not None else "tool_result",
            title,
            f"Analysis: {analysis_type} on {ds.name}",
            data={"figure": fig, "dataset_name": ds.name, "analysis_type": analysis_type, **result} if fig else
                 {"dataset_name": ds.name, "analysis_type": analysis_type, **result},
        )
        return vision_tool_content(json.dumps(to_json_safe(result)), fig), step, False


# ──────────────────────────────────────────────────────────────────────────────
# Chart construction
# ──────────────────────────────────────────────────────────────────────────────


def _build_figure(
    df: pd.DataFrame, dataset_name: str, chart_type: str, params: dict
) -> tuple[go.Figure | None, str, str]:
    log.debug(
        "_build_figure | dataset=%s chart_type=%r params=%s df_cols=%s",
        dataset_name, chart_type, params, list(df.columns)[:20],
    )
    try:
        if chart_type == "histogram":
            col = params.get("column")
            if not col or col not in df.columns:
                log.warning(
                    "_build_figure | histogram: column=%r not in dataset=%s (available: %s)",
                    col, dataset_name, list(df.columns)[:10],
                )
                return None, "", f"Column '{col}' not found."
            fig = px.histogram(
                df, x=col, nbins=int(params.get("bins", 30)),
                title=f"Distribution of {col}", template="plotly_white",
            )
            return fig, f"Histogram: {col}", f"Distribution of {col} in {dataset_name}"

        if chart_type == "bar":
            col = params.get("column")
            if not col or col not in df.columns:
                log.warning(
                    "_build_figure | bar: column=%r not in dataset=%s", col, dataset_name
                )
                return None, "", f"Column '{col}' not found."
            top_n = int(params.get("top_n", 20))
            vc = df[col].value_counts().head(top_n).reset_index()
            # Normalise column names across pandas versions:
            # pandas 1.x → ['index', col]; pandas 2.x → [col, 'count']
            vc.columns = ["value", "count"]
            fig = px.bar(
                vc, x="value", y="count",
                title=f"Value Counts: {col} (top {top_n})", template="plotly_white",
                labels={"value": col},
            )
            return fig, f"Bar: {col}", f"Top {top_n} value counts of {col} in {dataset_name}"

        if chart_type == "scatter":
            x_col = params.get("x_column")
            y_col = params.get("y_column")
            if not x_col or x_col not in df.columns:
                log.warning(
                    "_build_figure | scatter: x_column=%r not in dataset=%s", x_col, dataset_name
                )
                return None, "", f"x_column '{x_col}' not found."
            if not y_col or y_col not in df.columns:
                log.warning(
                    "_build_figure | scatter: y_column=%r not in dataset=%s", y_col, dataset_name
                )
                return None, "", f"y_column '{y_col}' not found."
            color_col = params.get("color_column")
            fig = px.scatter(
                df.head(2000), x=x_col, y=y_col,
                color=color_col if color_col and color_col in df.columns else None,
                title=f"Scatter: {x_col} vs {y_col}",
                template="plotly_white", opacity=0.6,
            )
            return (
                fig,
                f"Scatter: {x_col} vs {y_col}",
                f"Scatter of {x_col} vs {y_col} in {dataset_name}",
            )

        if chart_type == "correlation_heatmap":
            num = df.select_dtypes(include="number")
            if num.shape[1] < 2:
                return None, "", "Need ≥2 numeric columns."
            corr = num.corr()
            fig = px.imshow(
                corr, title=f"Correlation — {dataset_name}",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                text_auto=".2f", template="plotly_white",
            )
            return (
                fig,
                "Correlation Heatmap",
                f"Correlation of {num.shape[1]} numeric cols in {dataset_name}",
            )

        if chart_type == "box":
            col = params.get("column")
            if not col or col not in df.columns:
                log.warning(
                    "_build_figure | box: column=%r not in dataset=%s", col, dataset_name
                )
                return None, "", f"Column '{col}' not found."
            gb = params.get("group_by")
            fig = px.box(
                df, y=col, x=gb if gb and gb in df.columns else None,
                title=f"Box Plot: {col}", template="plotly_white",
            )
            label = f"Box: {col}" + (f" by {gb}" if gb and gb in df.columns else "")
            return fig, label, f"Box plot of {col} in {dataset_name}"

        if chart_type == "violin":
            col = params.get("column")
            if not col or col not in df.columns:
                log.warning(
                    "_build_figure | violin: column=%r not in dataset=%s", col, dataset_name
                )
                return None, "", f"Column '{col}' not found."
            if not pd.api.types.is_numeric_dtype(df[col]):
                log.warning(
                    "_build_figure | violin: column=%r is not numeric in dataset=%s", col, dataset_name
                )
                return None, "", f"Column '{col}' must be numeric for a violin plot."
            gb = params.get("group_by")
            fig = px.violin(
                df, y=col,
                x=gb if gb and gb in df.columns else None,
                box=True, points="outliers",
                title=f"Violin: {col}", template="plotly_white",
            )
            label = f"Violin: {col}" + (f" by {gb}" if gb and gb in df.columns else "")
            return fig, label, f"Violin of {col} in {dataset_name}"

        if chart_type == "pairplot":
            requested = params.get("columns") or []
            # Keep only existing, numeric columns (scatter matrix requires numeric).
            cols = [
                c for c in requested
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ][:4]
            if len(cols) < 2:
                # Fall back: auto-pick the first 4 numeric columns.
                cols = df.select_dtypes(include="number").columns.tolist()[:4]
            if len(cols) < 2:
                return None, "", "pairplot needs ≥2 numeric columns."

            color_col = params.get("color_column")
            valid_color = color_col if color_col and color_col in df.columns else None
            # Build data slice without duplicating the color column.
            plot_cols = list(dict.fromkeys(cols + ([valid_color] if valid_color and valid_color not in cols else [])))
            plot_df = df.head(2000)[plot_cols].copy()

            fig = px.scatter_matrix(
                plot_df,
                dimensions=cols,
                color=valid_color,
                title=f"Pairplot: {', '.join(cols)}",
                template="plotly_white",
            )
            # Hide upper triangle and diagonal to reduce visual clutter.
            fig.update_traces(
                diagonal=dict(visible=False),
                showupperhalf=False,
            )
            return (
                fig,
                f"Pairplot ({len(cols)} cols)",
                f"Pairplot of {cols} in {dataset_name}",
            )

        if chart_type == "missing_heatmap":
            mask = df.isnull()
            missing_cols = mask.columns[mask.any()].tolist()
            if not missing_cols:
                return None, "", "No missing values — nothing to chart."
            sample = mask[missing_cols].head(100).astype(int)
            fig = px.imshow(
                sample.T,
                title=f"Missing Values — {dataset_name} (first 100 rows)",
                color_continuous_scale=[[0, "#f0f0f0"], [1, "#e53e3e"]],
                labels={"x": "Row", "y": "Column", "color": "Missing"},
                template="plotly_white",
            )
            return (
                fig,
                "Missing Heatmap",
                f"Missing pattern in {len(missing_cols)} cols of {dataset_name}",
            )

        if chart_type == "line":
            x_col = params.get("x_column")
            y_cols = params.get("columns") or ([params.get("y_column")] if params.get("y_column") else None)
            if not x_col or x_col not in df.columns:
                return None, "", f"x_column '{x_col}' not found."
            if not y_cols:
                return None, "", "Provide params.y_column or params.columns for the line chart."
            y_cols = [c for c in y_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
            if not y_cols:
                return None, "", "No valid numeric y columns found."
            color_col = params.get("color_column")
            plot_df = df.sort_values(x_col).head(5000)
            if len(y_cols) == 1:
                fig = px.line(
                    plot_df, x=x_col, y=y_cols[0],
                    color=color_col if color_col and color_col in df.columns else None,
                    title=f"Line: {y_cols[0]} over {x_col}", template="plotly_white",
                )
            else:
                import plotly.graph_objects as _go
                fig = _go.Figure()
                for y in y_cols:
                    fig.add_trace(_go.Scatter(x=plot_df[x_col], y=plot_df[y], mode="lines", name=y))
                fig.update_layout(title=f"Line chart over {x_col}", template="plotly_white")
            label = f"Line: {y_cols} over {x_col}"
            return fig, label, label

        if chart_type == "qq_plot":
            import scipy.stats as stats_scipy
            col = params.get("column")
            if not col or col not in df.columns:
                return None, "", f"Column '{col}' not found."
            if not pd.api.types.is_numeric_dtype(df[col]):
                return None, "", f"Column '{col}' must be numeric for Q-Q plot."
            sample = df[col].dropna().values
            (osm, osr), (slope, intercept, _) = stats_scipy.probplot(sample, dist="norm")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(osm), y=list(osr), mode="markers", name="Quantiles"))
            fig.add_trace(go.Scatter(
                x=[float(min(osm)), float(max(osm))],
                y=[slope * float(min(osm)) + intercept, slope * float(max(osm)) + intercept],
                mode="lines", name="Normal line",
            ))
            fig.update_layout(
                title=f"Q-Q Plot: {col}", xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles", template="plotly_white",
            )
            return fig, f"Q-Q Plot: {col}", f"Normal Q-Q plot of {col} in {dataset_name}"

        log.warning("_build_figure | unknown chart_type=%r for dataset=%s", chart_type, dataset_name)
        return None, "", f"Unknown chart_type: {chart_type}"
    except Exception as exc:
        log.exception("_build_figure | exception chart_type=%r dataset=%s", chart_type, dataset_name)
        return None, "", f"Chart error: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Statistical / time-series analyses
# ──────────────────────────────────────────────────────────────────────────────


def _run_analysis(
    df: pd.DataFrame, dataset_name: str, analysis_type: str, params: dict
) -> tuple[dict, go.Figure | None, str]:
    """Run a named analysis. Returns (result_dict, optional_fig, title)."""
    try:
        if analysis_type == "class_balance":
            col = params.get("column")
            if not col or col not in df.columns:
                return {"error": f"params.column required and must exist. Available: {list(df.columns)[:20]}"}, None, ""
            counts = df[col].value_counts()
            imbalance = float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")
            result = {
                "column": col,
                "class_counts": counts.to_dict(),
                "n_classes": int(len(counts)),
                "imbalance_ratio": round(imbalance, 3),
                "is_imbalanced": imbalance > 3.0,
                "recommendation": (
                    "Apply SMOTE / resampling before training." if imbalance > 3.0
                    else "Class balance looks acceptable."
                ),
            }
            fig = px.bar(
                x=counts.index.astype(str).tolist(), y=counts.values.tolist(),
                title=f"Class Balance: {col} in {dataset_name}",
                labels={"x": col, "y": "count"}, template="plotly_white",
            )
            return result, fig, f"Class Balance: {col}"

        if analysis_type == "target_correlation":
            target = params.get("target_column")
            if not target or target not in df.columns:
                return {"error": f"params.target_column required. Available: {list(df.columns)[:20]}"}, None, ""
            num = df.select_dtypes(include="number")
            if target not in num.columns:
                return {"error": f"Target '{target}' must be numeric for correlation."}, None, ""
            corr = num.corr()[target].drop(target).sort_values(key=abs, ascending=False)
            result = {
                "target_column": target,
                "correlations": corr.round(4).to_dict(),
                "top_positive": corr[corr > 0].head(5).index.tolist(),
                "top_negative": corr[corr < 0].head(5).index.tolist(),
            }
            fig = px.bar(
                x=corr.index.tolist(), y=corr.values.tolist(),
                title=f"Feature Correlation with '{target}'",
                labels={"x": "Feature", "y": "Pearson r"}, template="plotly_white",
                color=corr.values.tolist(), color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
            )
            return result, fig, f"Target Correlation: {target}"

        if analysis_type == "mutual_information":
            target = params.get("target_column")
            if not target or target not in df.columns:
                return {"error": f"params.target_column required. Available: {list(df.columns)[:20]}"}, None, ""
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            from ..training import infer_task_type
            y = df[target]
            task = infer_task_type(y)
            num_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
            if not num_cols:
                return {"error": "No numeric feature columns for mutual information."}, None, ""
            X = df[num_cols].fillna(df[num_cols].median())
            mi_fn = mutual_info_classif if task == "classification" else mutual_info_regression
            mi = mi_fn(X, y, random_state=42)
            mi_series = pd.Series(mi, index=num_cols).sort_values(ascending=False)
            result = {
                "target_column": target,
                "task_type": task,
                "mutual_information": mi_series.round(4).to_dict(),
                "top_features": mi_series.head(10).index.tolist(),
            }
            fig = px.bar(
                x=mi_series.index.tolist(), y=mi_series.values.tolist(),
                title=f"Mutual Information with '{target}' ({task})",
                labels={"x": "Feature", "y": "MI score"}, template="plotly_white",
            )
            return result, fig, f"Mutual Information: {target}"

        if analysis_type == "normality_test":
            col = params.get("column")
            if not col or col not in df.columns:
                return {"error": f"params.column required. Available: {list(df.columns)[:20]}"}, None, ""
            if not pd.api.types.is_numeric_dtype(df[col]):
                return {"error": f"Column '{col}' must be numeric."}, None, ""
            import scipy.stats as sp_stats
            sample = df[col].dropna().values
            if len(sample) > 5000:
                sample = sample[:5000]
            result: dict = {
                "column": col,
                "n": int(len(sample)),
                "mean": float(pd.Series(sample).mean()),
                "std": float(pd.Series(sample).std()),
                "skewness": float(sp_stats.skew(sample)),
                "kurtosis": float(sp_stats.kurtosis(sample)),
            }
            if len(sample) <= 5000:
                try:
                    stat, p = sp_stats.shapiro(sample[:5000])
                    result["shapiro_stat"] = round(float(stat), 4)
                    result["shapiro_p"] = round(float(p), 6)
                    result["is_normal_shapiro"] = bool(p > 0.05)
                except Exception:
                    pass
            try:
                stat2, p2 = sp_stats.normaltest(sample)
                result["dagostino_stat"] = round(float(stat2), 4)
                result["dagostino_p"] = round(float(p2), 6)
                result["is_normal_dagostino"] = bool(p2 > 0.05)
            except Exception:
                pass
            result["recommendation"] = (
                "Distribution looks approximately normal." if result.get("is_normal_dagostino", True)
                else "Non-normal; consider log_transform, power_transform, or quantile_transform."
            )
            return result, None, f"Normality Test: {col}"

        if analysis_type == "vif":
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            num = df.select_dtypes(include="number").dropna()
            if num.shape[1] < 2:
                return {"error": "VIF needs ≥2 numeric columns."}, None, ""
            num = num.assign(__const=1.0)
            vif_data = {
                col: round(float(variance_inflation_factor(num.values, i)), 2)
                for i, col in enumerate(num.columns) if col != "__const"
            }
            vif_series = pd.Series(vif_data).sort_values(ascending=False)
            high_vif = vif_series[vif_series > 10].index.tolist()
            result = {
                "vif": vif_data,
                "high_vif_columns": high_vif,
                "recommendation": (
                    f"High multicollinearity detected in {high_vif}. "
                    "Consider drop_correlated or PCA." if high_vif
                    else "No severe multicollinearity detected (VIF ≤ 10 for all features)."
                ),
            }
            fig = px.bar(
                x=vif_series.index.tolist(), y=vif_series.values.tolist(),
                title=f"Variance Inflation Factor — {dataset_name}",
                labels={"x": "Feature", "y": "VIF"}, template="plotly_white",
            )
            fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="VIF=10 threshold")
            return result, fig, "VIF (Multicollinearity)"

        if analysis_type == "outlier_summary":
            import scipy.stats as sp_stats
            num = df.select_dtypes(include="number")
            if num.empty:
                return {"error": "No numeric columns for outlier summary."}, None, ""
            rows = []
            for col in num.columns:
                s = num[col].dropna()
                if len(s) == 0:
                    continue
                q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
                iqr = q3 - q1
                iqr_outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
                z = (s - s.mean()) / (s.std() + 1e-12)
                z_outliers = int((z.abs() > 3).sum())
                rows.append({
                    "column": col, "iqr_outliers": iqr_outliers,
                    "zscore_outliers": z_outliers, "total": len(s),
                    "iqr_pct": round(100 * iqr_outliers / len(s), 1),
                })
            result = {"outlier_summary": rows, "n_columns": len(rows)}
            return result, None, "Outlier Summary"

        if analysis_type == "seasonal_decompose":
            col = params.get("column")
            time_col = params.get("time_column")
            period = params.get("period")
            model = params.get("model", "additive")
            if not col or col not in df.columns:
                return {"error": f"params.column required. Available: {list(df.columns)[:20]}"}, None, ""
            if time_col and time_col in df.columns:
                work = df.sort_values(time_col).copy()
            else:
                work = df.copy()
            series = work[col].dropna()
            if len(series) < 4:
                return {"error": "Need ≥4 observations for seasonal decomposition."}, None, ""
            if period is None:
                period = min(12, len(series) // 2)
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose as _sd
                decomp = _sd(series, model=model, period=int(period), extrapolate_trend="freq")
            except Exception as exc:
                return {"error": f"seasonal_decompose failed: {exc}"}, None, ""

            trend_vals = decomp.trend.dropna().values.tolist()
            seasonal_vals = decomp.seasonal.values.tolist()
            residual_vals = decomp.resid.dropna().values.tolist()

            import numpy as _np
            seasonal_strength = (
                float(_np.var(seasonal_vals) / (_np.var(seasonal_vals) + _np.var(residual_vals) + 1e-12))
                if residual_vals else 0.0
            )
            result = {
                "column": col, "model": model, "period": int(period),
                "seasonal_strength": round(seasonal_strength, 4),
                "has_strong_seasonality": seasonal_strength > 0.3,
                "recommendation": (
                    f"Strong seasonality detected (strength={seasonal_strength:.2f}). "
                    "Use Fourier features, lag features at multiples of the period, "
                    "and ensure the Modeling Agent uses time_column for chronological split."
                    if seasonal_strength > 0.3 else
                    "Seasonality is weak — standard lag/rolling features may suffice."
                ),
            }
            fig = go.Figure()
            x_axis = list(range(len(series)))
            fig.add_trace(go.Scatter(x=x_axis, y=series.values.tolist(), name="Observed", mode="lines"))
            fig.add_trace(go.Scatter(x=list(range(len(decomp.trend.dropna()))), y=trend_vals, name="Trend", mode="lines"))
            fig.add_trace(go.Scatter(x=x_axis, y=seasonal_vals, name="Seasonal", mode="lines"))
            fig.add_trace(go.Scatter(x=list(range(len(decomp.resid.dropna()))), y=residual_vals, name="Residual", mode="lines"))
            fig.update_layout(title=f"Seasonal Decomposition: {col} (period={period})", template="plotly_white")
            return result, fig, f"Seasonal Decomposition: {col}"

        if analysis_type == "stationarity_test":
            col = params.get("column")
            if not col or col not in df.columns:
                return {"error": f"params.column required. Available: {list(df.columns)[:20]}"}, None, ""
            if not pd.api.types.is_numeric_dtype(df[col]):
                return {"error": f"Column '{col}' must be numeric."}, None, ""
            series = df[col].dropna()
            if len(series) < 8:
                return {"error": "Need ≥8 observations for stationarity tests."}, None, ""
            try:
                from statsmodels.tsa.stattools import adfuller, kpss
                adf_stat, adf_p, adf_lags, _, adf_cv, _ = adfuller(series, autolag="AIC")
                try:
                    kpss_stat, kpss_p, kpss_lags, kpss_cv = kpss(series, regression="c", nlags="auto")
                    kpss_info = {
                        "kpss_stat": round(float(kpss_stat), 4),
                        "kpss_p": round(float(kpss_p), 4),
                        "kpss_is_stationary": bool(kpss_p > 0.05),
                    }
                except Exception:
                    kpss_info = {}
                result = {
                    "column": col,
                    "adf_stat": round(float(adf_stat), 4),
                    "adf_p": round(float(adf_p), 6),
                    "adf_is_stationary": bool(adf_p < 0.05),
                    **kpss_info,
                }
                adf_stat_flag = result["adf_is_stationary"]
                kpss_stat_flag = kpss_info.get("kpss_is_stationary", True)
                if adf_stat_flag and kpss_stat_flag:
                    verdict = "Stationary (both ADF and KPSS agree). No differencing needed."
                elif not adf_stat_flag and not kpss_stat_flag:
                    verdict = "Non-stationary. Consider differencing or using dense_panel + lags."
                else:
                    verdict = "Conflicting results — possibly trend-stationary. Inspect the line chart."
                result["verdict"] = verdict
            except Exception as exc:
                return {"error": f"Stationarity test failed: {exc}"}, None, ""
            return result, None, f"Stationarity Test: {col}"

        if analysis_type == "acf_pacf":
            col = params.get("column")
            if not col or col not in df.columns:
                return {"error": f"params.column required. Available: {list(df.columns)[:20]}"}, None, ""
            if not pd.api.types.is_numeric_dtype(df[col]):
                return {"error": f"Column '{col}' must be numeric."}, None, ""
            series = df[col].dropna()
            if len(series) < 4:
                return {"error": "Need ≥4 observations for ACF/PACF."}, None, ""
            nlags = min(int(params.get("nlags", 40)), len(series) // 2 - 1)
            nlags = max(nlags, 1)
            try:
                from statsmodels.tsa.stattools import acf, pacf
                acf_vals = acf(series, nlags=nlags, fft=True).tolist()
                pacf_vals = pacf(series, nlags=nlags, method="ols").tolist()
            except Exception as exc:
                return {"error": f"ACF/PACF failed: {exc}"}, None, ""
            conf = 1.96 / (len(series) ** 0.5)
            significant_acf = [i for i, v in enumerate(acf_vals[1:], 1) if abs(v) > conf]
            significant_pacf = [i for i, v in enumerate(pacf_vals[1:], 1) if abs(v) > conf]
            result = {
                "column": col, "nlags": nlags,
                "acf": [round(v, 4) for v in acf_vals],
                "pacf": [round(v, 4) for v in pacf_vals],
                "significant_acf_lags": significant_acf[:12],
                "significant_pacf_lags": significant_pacf[:12],
                "confidence_interval": round(conf, 4),
                "recommendation": (
                    f"Significant ACF lags: {significant_acf[:6]}. "
                    f"Significant PACF lags: {significant_pacf[:6]}. "
                    "Include these as lag features. PACF cut-off suggests AR order."
                ),
            }
            lags = list(range(len(acf_vals)))
            fig = go.Figure()
            fig.add_bar(x=lags, y=acf_vals, name="ACF", marker_color="steelblue")
            fig.add_trace(go.Scatter(x=lags, y=[conf] * len(lags), mode="lines",
                                     line=dict(dash="dash", color="red"), name="95% CI"))
            fig.add_trace(go.Scatter(x=lags, y=[-conf] * len(lags), mode="lines",
                                     line=dict(dash="dash", color="red"), showlegend=False))
            fig.update_layout(title=f"ACF/PACF: {col}", template="plotly_white",
                              xaxis_title="Lag", yaxis_title="Correlation")
            return result, fig, f"ACF/PACF: {col}"

        return {"error": f"Unknown analysis_type: {analysis_type}"}, None, ""
    except Exception as exc:
        log.exception("_run_analysis | exception analysis_type=%r dataset=%s", analysis_type, dataset_name)
        return {"error": f"Analysis error: {exc}"}, None, ""
