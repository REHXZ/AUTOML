"""Feature Engineering Agent: creates derived datasets via transformations."""

from __future__ import annotations

import json
import logging
from typing import Any, Generator

import numpy as np
import pandas as pd

from ..ingestion import load_dataset
from backend.server.logging_setup import configure_logging
from backend.services.project_store import DatasetInfo
from .base import AgentContext, AutopilotStep, BaseAgent, to_json_safe

configure_logging()
log = logging.getLogger(__name__)

# Per-operation parameter keys the LLM sometimes sends at the top level of the
# tool args instead of nested under "params". We auto-promote them to params
# (with a WARNING) so the operation still runs.
_FE_PARAM_KEYS = frozenset({
    "threshold",          # drop_high_missing, drop_correlated, select_from_model, zscore
    "columns",            # select/drop_columns, one_hot_encode, log_transform, polynomial_features, scalers, encoders
    "max_unique",         # one_hot_encode
    "column",             # bin_numeric, target_log_transform, lag/lead/rolling, clip, cyclical, label/fourier
    "bins",               # bin_numeric
    "pairs",              # interaction_features
    "degree",             # polynomial_features
    "group_by",           # groupby_aggregate, lag/lead/rolling, dense_panel
    "aggregations",       # groupby_aggregate
    "rename",             # rename_columns
    "time_column",        # lag/lead/rolling, dense_panel, fourier_features
    "lags",               # create_lag_features
    "leads",              # create_lead_target
    "windows",            # create_rolling_features
    "agg",                # create_rolling_features
    "freq",               # dense_panel
    "fill_value",         # dense_panel, constant_impute
    "code",               # execute_python
    # ── extended operations ──
    "target_column",      # target_encode, select_*, rfe, pca, resamplers, outlier ops
    "method",             # power_transform
    "output_distribution",  # quantile_transform
    "n_quantiles",        # quantile_transform
    "min", "max",         # clip_values
    "lower", "upper",     # winsorize
    "n_neighbors",        # knn_impute, resamplers (k_neighbors alias handled separately)
    "k_neighbors",        # smote family
    "max_iter",           # iterative_impute
    "normalize",          # frequency_encode
    "smoothing",          # target_encode
    "period",             # cyclical_encode, fourier_features
    "order",              # fourier_features
    "k",                  # select_k_best
    "mutual_info",        # select_k_best
    "n_features_to_select",  # rfe_select
    "n_components",       # pca
    "contamination",      # isolation_forest_outliers
    "exclude", "exclude_columns",  # scalers / outlier ops
    "n_features",                 # hash_encode
})
# Operations that require params at all — used to escalate the missing-params warning.
_OPS_REQUIRING_PARAMS = frozenset({
    "select_columns", "drop_columns", "one_hot_encode", "log_transform",
    "bin_numeric", "interaction_features", "polynomial_features",
    "target_log_transform", "groupby_aggregate", "rename_columns",
    "dense_panel", "create_lag_features", "create_rolling_features",
    "create_lead_target", "execute_python",
    # ── extended operations that need at least one param ──
    "clip_values", "label_encode", "target_encode", "cyclical_encode",
    "fourier_features", "datetime_parse", "select_k_best",
    "select_from_model", "rfe_select", "smote", "borderline_smote",
    "adasyn", "random_oversample", "random_undersample", "smote_tomek",
    "smote_enn",
})


_SYSTEM_PROMPT = """\
# Role & Objective
You are the Feature Engineering Agent — a senior ML engineer specialising
in feature design, data cleaning, and transformations that maximise
downstream model signal. You operate on datasets and produce improved ones.

════════════════════════════════════════════════════════════════════════
# REASONING PROTOCOL — THINK BEFORE EVERY OPERATION

Before each create_derived_dataset call, write a reasoning step:
  "Current dataset: [id], shape: [N rows × M cols].
   I will apply [operation] because [evidence: EDA finding, Review critique,
   or domain knowledge].
   Expected effect: [specific outcome: cleaner distribution, less leakage,
   more signal, etc.].
   Parameters: [key params and why I chose them]."

After creating each dataset, validate by calling inspect_dataset (via the
Modeling Agent or by reading profile results) and confirm the operation
had the expected effect. If not, adjust and retry.

════════════════════════════════════════════════════════════════════════
# OPERATION ORDERING RULES (apply in this sequence)

  1. CLEANING & IMPUTATION (must come first)
     drop_high_missing → drop_constant → drop_correlated → drop_duplicates
     → impute_missing (or knn_impute) → add_missing_indicators

  2. OUTLIER HANDLING (before scaling)
     winsorize → zscore_outlier_removal → isolation_forest_outliers

  3. ENCODING (before scaling, after imputation)
     datetime_parse → encode_dates → cyclical_encode → one_hot_encode /
     ordinal_encode / target_encode / hash_encode / frequency_encode

  4. SCALING & TRANSFORMS (after encoding)
     log_transform / target_log_transform → standard_scale / robust_scale /
     power_transform / quantile_transform

  5. FEATURE GENERATION (after cleaning and encoding)
     interaction_features → polynomial_features → groupby_aggregate →
     create_lag_features → create_rolling_features → create_lead_target

  6. FEATURE SELECTION (last step before modeling)
     select_k_best → select_from_model → rfe_select → pca

  7. CLASS RESAMPLING (very last — must have NO NaNs, NO raw categoricals)
     smote / smote_tomek / smote_enn / adasyn / random_oversample

════════════════════════════════════════════════════════════════════════
# PROBLEM-TYPE PIPELINES

## Classification Pipeline (tabular, imbalanced)
  1. drop_high_missing (threshold 0.5)
  2. drop_constant
  3. impute_missing + add_missing_indicators (for cols with > 5 % missing)
  4. encode_dates on any datetime columns
  5. one_hot_encode (cardinality ≤ 20) OR target_encode (cardinality > 20)
  6. log_transform on right-skewed numeric features (skewness > 1.5)
  7. standard_scale OR robust_scale (robust if outliers remain)
  8. [Optional] select_from_model to prune weak features
  9. smote_tomek (if imbalance > 5:1)

## Regression Pipeline (tabular)
  1. drop_high_missing (threshold 0.5)
  2. drop_constant
  3. impute_missing + add_missing_indicators
  4. encode_dates on any datetime columns
  5. one_hot_encode OR target_encode categoricals
  6. target_log_transform if target skewness > 1.5 (IN PLACE)
  7. log_transform on right-skewed features
  8. robust_scale (preferred for regression with outliers)
  9. [Optional] drop_correlated (threshold 0.9)
  10. [Optional] select_from_model

## Forecasting Pipeline (time-series)
  1. groupby_aggregate → aggregate to forecast period (monthly/weekly)
  2. dense_panel → fill missing (group × time) combinations with fill_value=0
  3. create_lag_features → lags [1, 2, 3, 6, 12] (for monthly data)
  4. create_rolling_features → windows [3, 6, 12], agg "mean" and "std"
  5. encode_dates on the time column
  6. cyclical_encode month (period=12) and day-of-week (period=7)
  7. target_encode for high-cardinality group columns
  8. create_lead_target → leads [1, 3] for short-term and medium-term targets
  DO NOT apply class resampling or PCA to time-series datasets.

## NLP / Text Pipeline
  1. text_clean on all text columns (lowercase, strip punctuation)
  2. text_length_features (adds char_count, word_count features)
  3. tfidf_vectorize (max_features=100, ngram_range=[1,2]) OR
     lsa_text (n_components=20, for dense topic features)
  4. Drop the original text column after vectorization

════════════════════════════════════════════════════════════════════════
# PARAMETER SELECTION GUIDE

| Operation             | Key Parameter       | Recommended Value                              |
|-----------------------|---------------------|------------------------------------------------|
| drop_high_missing     | threshold           | 0.5 (drop if > 50% missing)                   |
| knn_impute            | n_neighbors         | 5 (default); 3 for small datasets              |
| one_hot_encode        | (cardinality check) | Use when unique values ≤ 20                    |
| target_encode         | smoothing           | 10 (default); increase to 20 for small groups |
| hash_encode           | n_features          | 32-64 for < 1k cats; 128 for > 1k cats        |
| log_transform         | columns             | Any column with skewness > 1.5                 |
| robust_scale          | (outliers present?) | Use when outlier rate > 5 %                    |
| standard_scale        | (no outliers)       | Use when distribution is roughly Gaussian      |
| create_lag_features   | lags                | [1,2,3,6,12] for monthly; [1,7,14,28] for daily|
| create_rolling_features| windows            | [3,6,12] for monthly; [7,14,30] for daily      |
| smote                 | k_neighbors         | 5 (default); 3 if minority class has < 20 rows|
| select_k_best         | k                   | Start with 20; reduce if still overfitting     |
| cyclical_encode       | period              | 12 for month, 7 for dow, 24 for hour           |

CRITICAL CALLING CONVENTION
  Every per-operation argument (group_by, aggregations, columns, column,
  bins, lags, leads, windows, time_column, freq, fill_value, rename, etc.)
  MUST be nested INSIDE the "params" object. Top-level keys are ignored.
  Example:
    {"source_dataset_id":"abc","new_name":"monthly_demand",
     "operation":"groupby_aggregate",
     "params":{"group_by":["request_month","shimano_part_no"],
               "aggregations":{"qty":"sum","shimano_order_no":"nunique"}}}

OPERATIONS AVAILABLE
────────────────────────────────────────────────────────
CLEANING
  • drop_high_missing   — drop cols with > params.threshold missing (default 0.5)
  • drop_duplicates     — remove duplicate rows
  • select_columns      — keep only params.columns
  • drop_columns        — remove params.columns
  • drop_constant       — drop zero-variance / single-value columns (no params needed)
  • drop_correlated     — drop one of each pair correlated > params.threshold (default 0.95)
  • filter_outliers     — IQR outlier removal on numeric cols (1.5 IQR)
  • impute_missing      — fill numeric with median, categorical with mode
  • constant_impute     — fill params.columns with params.fill_value (default 0)
  • knn_impute          — KNN-based imputation; params.columns, params.n_neighbors (default 5)
  • iterative_impute    — MICE / IterativeImputer; params.columns, params.max_iter (default 10)
  • add_missing_indicators — add {col}_was_missing boolean flags; params.columns optional

OUTLIER HANDLING
  • winsorize           — clip to percentile range; params.lower (default 0.01), params.upper (0.99)
  • clip_values         — clip params.column to [params.min, params.max]
  • zscore_outlier_removal — drop rows with |z| > params.threshold (default 3.0)
  • isolation_forest_outliers — remove anomalies; params.contamination (default "auto")

SCALING & NUMERIC TRANSFORMS
  • standard_scale      — z-score (StandardScaler); params.columns (optional, default all numeric)
  • minmax_scale        — [0,1] scale (MinMaxScaler); params.columns
  • robust_scale        — median/IQR scale, outlier-robust; params.columns
  • max_abs_scale       — divide by max absolute value; params.columns
  • power_transform     — Yeo-Johnson or Box-Cox; params.method ("yeo-johnson"|"box-cox")
  • quantile_transform  — rank-based; params.output_distribution ("normal"|"uniform")
  • log_transform       — log1p params.columns (adds new cols, preserves originals)
  • target_log_transform— log1p of params.column IN PLACE (regression skew fix)
  • bin_numeric         — quantile-bin params.column into params.bins bins

ENCODING
  • encode_dates        — expand datetime → year/month/day/dayofweek/quarter
  • datetime_parse      — coerce params.columns to proper datetime dtype
  • one_hot_encode      — one-hot params.columns (or auto low-cardinality)
  • ordinal_encode      — integer-code categoricals in params.columns
  • label_encode        — integer-code a single params.column
  • frequency_encode    — replace categories with their frequency; params.normalize (default True)
  • target_encode       — smoothed mean-of-target; params.target_column, params.smoothing (default 10)
  • hash_encode         — feature hashing for very high cardinality categoricals
                          params.columns, params.n_features (default 64)
                          USE when one_hot_encode would produce thousands of columns.
  • cyclical_encode     — sin/cos encoding for cyclic features (month, hour, day-of-week)
                          params.column, params.period (e.g. 12 for month, 24 for hour)
  • fourier_features    — sin/cos harmonics for any periodic column
                          params.column, params.period, params.order (default 3)
  • interaction_features— multiplicative params.pairs: [[a,b], ...]
  • polynomial_features — squared/cubed of params.columns (params.degree)
  • rename_columns      — params.rename: {"old":"new", ...}

FEATURE SELECTION & DIMENSIONALITY REDUCTION
  • select_k_best       — keep top-k features by F-test or mutual info
                          params.target_column, params.k (default 10),
                          params.mutual_info (default False)
  • select_from_model   — RandomForest importance threshold
                          params.target_column, params.threshold ("median"|"mean"|float)
  • rfe_select          — Recursive Feature Elimination
                          params.target_column, params.n_features_to_select (default 10)
  • pca                 — Principal Component Analysis
                          params.n_components (default 5), params.target_column (excluded)

AGGREGATION
  • groupby_aggregate   — params.group_by + params.aggregations
                          Aggs: count|nunique|sum|mean|min|max|first|last
                          USE for monthly/weekly rollups of transactional rows.

CLASS IMBALANCE (classification targets only — impute first, no NaNs allowed)
  • smote               — SMOTE; auto-switches to SMOTENC when categoricals present
                          params.target_column, params.k_neighbors (default 5)
  • borderline_smote    — focus oversampling on the decision border (numeric features only)
                          params.target_column, params.k_neighbors
  • adasyn              — adaptive oversampling; params.target_column, params.k_neighbors
  • random_oversample   — random minority oversampling; params.target_column
  • random_undersample  — random majority undersampling; params.target_column
  • smote_tomek         — SMOTE + Tomek-link cleaning (combined); params.target_column
  • smote_enn           — SMOTE + Edited-Nearest-Neighbours (combined); params.target_column

  WHEN TO USE RESAMPLING: when EDA or Review flags class imbalance (one class
  is much smaller). SMOTE is the default choice. Use smote_tomek or smote_enn
  for a cleaner boundary. Use random_undersample when the majority class is so
  large that speed matters. Always resample AFTER other cleaning/encoding steps.

Time-series feature engineering (REQUIRED for proper forecasting):
  • dense_panel         — Fill missing (group × time) combinations so every
                          SKU has a row for every month, missing values
                          filled with params.fill_value (default 0).
                          params.time_column, params.group_by (list),
                          params.freq ("M"|"W"|"D", default "M"),
                          params.fill_value (default 0).
                          Run this BEFORE lag/rolling/lead features.
  • create_lag_features — Shift past values within group (backwards lag).
                          params.column (column to lag — usually the target),
                          params.lags (list of ints, e.g. [1,2,3,6,12]),
                          params.group_by (list, e.g. ["shimano_part_no"]),
                          params.time_column (sort by this before shifting).
                          Creates {column}_lag_{n} for each n.
  • create_rolling_features — Rolling window aggregates within group.
                          params.column, params.windows (e.g. [3,6,12]),
                          params.group_by, params.time_column,
                          params.agg ("mean"|"sum"|"std"|"min"|"max",
                                      default "mean").
                          Creates {column}_roll_{window}_{agg}.
                          IMPORTANT: rolling is computed on LAGGED values
                          (shift(1) before rolling) to prevent leakage.
  • create_lead_target  — Shift target FORWARD to create forecasting targets.
                          params.column (the column to lead),
                          params.leads (list of ints, e.g. [1,3]),
                          params.group_by, params.time_column.
                          Creates {column}_lead_{n} = next n-th period value.
                          This is how you build t+1 / t+3 forecasting targets:
                            qty_lead_1 → predict next month's qty
                            qty_lead_3 → predict qty 3 months ahead
                          After this, Modeling Agent trains with
                          target_column="qty_lead_1" using time_column for
                          chronological backtest.

Custom code:
  • execute_python — Run arbitrary Python on the dataframe.
                     params.code: Python string. 'df' is the input DataFrame.
                     Mutate or reassign 'df' and it will be saved as the output.
                     Only pandas (pd) and numpy (np) are pre-imported.
                     Example: "df['ratio'] = df['a'] / df['b'].replace(0, np.nan)"
                     Use for any transformation not covered by the operations above.

TYPICAL FORECASTING PIPELINE
  1. groupby_aggregate  → monthly (or weekly) panel
  2. dense_panel        → zero-fill missing SKU-month combinations
  3. create_lag_features→ qty_lag_1, qty_lag_3, qty_lag_12 (seasonality)
  4. create_rolling_features→ qty_roll_3_mean, qty_roll_12_mean (trend)
  5. create_lead_target → qty_lead_1, qty_lead_3 (the actual targets)
  6. Modeling Agent then trains on the result with time_column set so
     the holdout is the last N% of months, not a random split.

After each transformation, record a brief note explaining WHY it should help.
When you have produced a coherent feature set (or set of alternatives), call
done(summary) with the created dataset IDs and your rationale.
"""


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "create_derived_dataset",
                "description": (
                    "Apply a single transformation and register the result as a new dataset. "
                    "CRITICAL: per-operation arguments (group_by, aggregations, columns, rename, "
                    "column, bins, pairs, degree, threshold, etc.) MUST be nested INSIDE the "
                    "'params' object — NOT at the top level. "
                    "Example for groupby_aggregate: "
                    "{\"source_dataset_id\":\"abc\",\"new_name\":\"monthly_demand\","
                    "\"operation\":\"groupby_aggregate\","
                    "\"params\":{\"group_by\":[\"request_month\",\"shimano_part_no\"],"
                    "\"aggregations\":{\"qty\":\"sum\",\"shimano_order_no\":\"nunique\"}},"
                    "\"rationale\":\"...\"}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_dataset_id": {"type": "string"},
                        "new_name": {"type": "string"},
                        "operation": {
                            "type": "string",
                            "enum": [
                                "drop_high_missing", "drop_duplicates",
                                "select_columns", "drop_columns",
                                "filter_outliers", "encode_dates",
                                "one_hot_encode", "log_transform",
                                "impute_missing", "bin_numeric",
                                "interaction_features", "polynomial_features",
                                "target_log_transform",
                                "groupby_aggregate", "rename_columns",
                                "dense_panel", "create_lag_features",
                                "create_rolling_features", "create_lead_target",
                                "execute_python",
                                # ── scaling / numeric transforms ──
                                "standard_scale", "minmax_scale", "robust_scale",
                                "max_abs_scale", "power_transform",
                                "quantile_transform", "clip_values", "winsorize",
                                # ── imputation ──
                                "constant_impute", "knn_impute",
                                "iterative_impute", "add_missing_indicators",
                                # ── encoding ──
                                "ordinal_encode", "label_encode",
                                "frequency_encode", "target_encode",
                                "hash_encode",
                                "cyclical_encode", "fourier_features",
                                "datetime_parse",
                                # ── cleaning ──
                                "drop_constant", "drop_correlated",
                                # ── selection / reduction ──
                                "select_k_best", "select_from_model",
                                "rfe_select", "pca",
                                # ── outliers ──
                                "zscore_outlier_removal",
                                "isolation_forest_outliers",
                                # ── class-imbalance resampling ──
                                "smote", "borderline_smote", "adasyn",
                                "random_oversample", "random_undersample",
                                "smote_tomek", "smote_enn",
                            ],
                        },
                        "params": {"type": "object"},
                        "rationale": {
                            "type": "string",
                            "description": "Why you're making this transformation.",
                        },
                    },
                    "required": ["source_dataset_id", "new_name", "operation"],
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
                "name": "done",
                "description": "Finish feature engineering. Report the datasets you built.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "created_dataset_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "rationale": {"type": "string"},
                        "next_moves": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["rationale"],
                },
            },
        },
    ]


class FeatureEngineeringAgent(BaseAgent):
    name = "feature_engineering"
    display_name = "Feature Engineering Agent"

    def __init__(self, client, deployment: str, context: AgentContext) -> None:
        super().__init__(client, deployment, context)
        self._summary: dict[str, Any] = {}
        self._created_ids: list[str] = []

    def run(
        self, instructions: str
    ) -> Generator[AutopilotStep, list[str] | None, dict[str, Any]]:
        yield self._step(
            "agent_start",
            "Feature Engineering Agent dispatched",
            instructions or "(explore beneficial transformations)",
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
            "Produce one or more derived datasets that should improve modelling."
        )

        yield from self.run_llm_loop(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=_tools(),
            dispatch=self._dispatch,
            max_iterations=30,
            thought_title="Feature Engineering — Reasoning",
        )

        if self._created_ids:
            self._summary.setdefault("created_dataset_ids", self._created_ids)

        yield self._step("agent_end", "Feature Engineering finished", "")
        return self._summary or {"rationale": "FE agent ended without summary."}

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self, name: str, args: dict, tool_call_id: str
    ) -> tuple[str | None, AutopilotStep | None, bool]:
        if name == "create_derived_dataset":
            return self._create_derived(args)
        if name == "record_finding":
            text = (args.get("text") or "").strip()
            if text:
                self._ctx.notebook.append(f"[FE] {text}")
            return (
                json.dumps({"recorded": True}),
                self._step("observation", "FE finding", text),
                False,
            )
        if name == "done":
            self._summary = to_json_safe(args)
            return json.dumps({"status": "noted"}), None, True
        return json.dumps({"error": f"Unknown tool: {name}"}), None, False

    # ------------------------------------------------------------------
    # Derived dataset operations
    # ------------------------------------------------------------------

    def _create_derived(self, args: dict) -> tuple[str, AutopilotStep | None, bool]:
        source = self._ctx.find_dataset(args.get("source_dataset_id", ""))
        if source is None:
            log.warning("create_derived | source_dataset_id=%r not found", args.get("source_dataset_id"))
            return (
                json.dumps({"error": f"Source dataset '{args.get('source_dataset_id')}' not found."}),
                None,
                False,
            )
        loaded = load_dataset(source.file_path, source.table_name)
        df = loaded.dataframe.copy()
        operation = args.get("operation", "")
        params = args.get("params") or {}
        new_name = args.get("new_name") or f"{source.name}_{operation}"
        rationale = args.get("rationale", "")

        # Flat-params fallback: the LLM sometimes sends per-operation keys
        # (group_by, aggregations, columns, rename, etc.) at the top level of
        # the tool args instead of nested under "params". Auto-promote them so
        # the operation still runs, and warn so we can track the schema drift.
        if not params:
            flat = {k: v for k, v in args.items() if k in _FE_PARAM_KEYS}
            if flat:
                log.warning(
                    "create_derived | flat-params detected for operation=%r — "
                    "LLM sent %s at top level instead of nested under 'params'; promoting.",
                    operation, list(flat),
                )
                params = flat

        if not params and operation in _OPS_REQUIRING_PARAMS:
            log.warning(
                "create_derived | operation=%r requires params but received none — "
                "full args: %s",
                operation, args,
            )

        log.info(
            "create_derived | source=%s operation=%s params=%s new_name=%s",
            source.name, operation, params, new_name,
        )
        try:
            df, detail = _apply_operation(df, operation, params)
        except Exception as exc:
            log.error("create_derived | FAILED operation=%s source=%s error=%s", operation, source.name, exc)
            return json.dumps({"error": f"Operation failed: {exc}"}), None, False

        if df is None:
            log.warning("create_derived | operation=%s returned None: %s", operation, detail)
            return json.dumps({"error": detail}), None, False
        if df.empty:
            log.warning("create_derived | operation=%s produced empty dataset", operation)
            return json.dumps({"error": "Derived dataset is empty."}), None, False

        csv_bytes = df.to_csv(index=False).encode()
        filename = f"{new_name.replace(' ', '_')}.csv"
        saved = self._ctx.store.save_dataset_file(
            self._ctx.project_id, filename, csv_bytes
        )
        ds_info = self._ctx.store.register_dataset(
            self._ctx.project_id,
            name=new_name,
            source_name=filename,
            source_type="csv",
            file_path=str(saved),
            row_count=int(len(df)),
            column_count=int(len(df.columns)),
        )
        self._ctx.new_datasets.append(ds_info)
        self._created_ids.append(ds_info.id)
        if rationale:
            self._ctx.notebook.append(f"[FE] {new_name}: {rationale}")
        log.info(
            "create_derived | OK operation=%s new_dataset=%s id=%s rows=%d cols=%d",
            operation, new_name, ds_info.id, len(df), len(df.columns),
        )

        step = self._step(
            "new_dataset",
            f"New Dataset: {new_name}",
            f"{detail} → {len(df)} rows × {len(df.columns)} cols",
            data={
                "dataset_id": ds_info.id,
                "rows": int(len(df)),
                "cols": int(len(df.columns)),
                "operation": operation,
                "rationale": rationale,
            },
        )
        result = {
            "dataset_id": ds_info.id,
            "name": new_name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "detail": detail,
        }
        return json.dumps(result), step, False


def _apply_operation(
    df: pd.DataFrame, operation: str, params: dict
) -> tuple[pd.DataFrame | None, str]:
    if operation == "drop_high_missing":
        threshold = float(params.get("threshold", 0.5))
        before = df.shape[1]
        df = df.loc[:, df.isnull().mean() <= threshold]
        return df, f"Dropped {before - df.shape[1]} cols with >{threshold * 100:.0f}% missing"

    if operation == "drop_duplicates":
        before = len(df)
        df = df.drop_duplicates()
        return df, f"Removed {before - len(df)} duplicate rows"

    if operation == "select_columns":
        cols = [c for c in params.get("columns", []) if c in df.columns]
        if not cols:
            return None, "None of the specified columns exist."
        return df[cols], f"Selected {len(cols)} columns"

    if operation == "drop_columns":
        cols = [c for c in params.get("columns", []) if c in df.columns]
        if not cols:
            return None, "None of the specified columns exist."
        return df.drop(columns=cols), f"Dropped {len(cols)} columns"

    if operation == "filter_outliers":
        before = len(df)
        for col in df.select_dtypes(include="number").columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
        return df, f"Removed {before - len(df)} outlier rows (IQR)"

    if operation == "encode_dates":
        # Try to coerce object cols that look like dates first.
        for col in df.select_dtypes(include="object").columns:
            try:
                converted = pd.to_datetime(df[col], errors="raise")
                df[col] = converted
            except Exception:
                continue
        dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
        for col in dt_cols:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_quarter"] = df[col].dt.quarter
        df = df.drop(columns=dt_cols)
        return df, f"Expanded {len(dt_cols)} datetime col(s)"

    if operation == "one_hot_encode":
        cols = params.get("columns") or []
        if not cols:
            cols = [
                c for c in df.select_dtypes(include=["object", "category"]).columns
                if df[c].nunique(dropna=True) <= int(params.get("max_unique", 15))
            ]
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return None, "No suitable columns to one-hot encode."
        df = pd.get_dummies(df, columns=cols, drop_first=False, dummy_na=False)
        return df, f"One-hot encoded {len(cols)} column(s)"

    if operation == "log_transform":
        cols = [c for c in (params.get("columns") or []) if c in df.columns]
        if not cols:
            return None, "No valid columns for log_transform."
        for col in cols:
            df[f"{col}_log1p"] = np.log1p(df[col].clip(lower=0))
        return df, f"Added log1p of {len(cols)} column(s)"

    if operation == "impute_missing":
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    mode = df[col].mode()
                    if not mode.empty:
                        df[col] = df[col].fillna(mode.iloc[0])
        return df, "Imputed missing values (median / mode)"

    if operation == "bin_numeric":
        col = params.get("column")
        bins = int(params.get("bins", 5))
        if col not in df.columns:
            return None, f"Column '{col}' not found."
        df[f"{col}_bin"] = pd.qcut(
            df[col], q=bins, labels=False, duplicates="drop"
        )
        return df, f"Binned {col} into {bins} quantile bins"

    if operation == "interaction_features":
        pairs = params.get("pairs") or []
        added = 0
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            a, b = pair
            if a in df.columns and b in df.columns:
                if pd.api.types.is_numeric_dtype(df[a]) and pd.api.types.is_numeric_dtype(df[b]):
                    df[f"{a}_x_{b}"] = df[a] * df[b]
                    added += 1
        if added == 0:
            return None, "No valid numeric pairs to multiply."
        return df, f"Added {added} interaction feature(s)"

    if operation == "polynomial_features":
        cols = [c for c in (params.get("columns") or []) if c in df.columns]
        degree = int(params.get("degree", 2))
        added = 0
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                for d in range(2, degree + 1):
                    df[f"{col}_pow{d}"] = df[col] ** d
                    added += 1
        if added == 0:
            return None, "No valid numeric columns for polynomial_features."
        return df, f"Added {added} polynomial feature(s)"

    if operation == "target_log_transform":
        col = params.get("column")
        if not col or col not in df.columns:
            return None, "Target column missing for target_log_transform."
        if not pd.api.types.is_numeric_dtype(df[col]):
            return None, f"Column '{col}' is not numeric."
        df[col] = np.log1p(df[col].clip(lower=0))
        return df, f"Applied log1p to target '{col}'"

    if operation == "groupby_aggregate":
        group_cols: list = params.get("group_by") or []
        agg_spec: dict = params.get("aggregations") or {}
        if not group_cols:
            return None, (
                "group_by must list at least one column. "
                "REMINDER: pass group_by and aggregations NESTED inside the 'params' object, e.g. "
                '{"operation":"groupby_aggregate", '
                '"params":{"group_by":["request_month","shimano_part_no"], '
                '"aggregations":{"qty":"sum","shimano_order_no":"nunique"}}}'
            )
        if not agg_spec:
            return None, (
                "aggregations dict required NESTED inside 'params' — e.g. "
                '"params":{"group_by":[...], "aggregations":{"qty":"sum","order_id":"nunique"}}'
            )
        missing_group = [c for c in group_cols if c not in df.columns]
        if missing_group:
            return None, (
                f"group_by columns not found: {missing_group}. "
                f"Available: {list(df.columns)[:20]}"
            )
        valid_agg = {col: func for col, func in agg_spec.items() if col in df.columns}
        if not valid_agg:
            return None, (
                f"None of the aggregation columns exist. "
                f"Available columns: {list(df.columns)[:20]}"
            )
        grouped = df.groupby(group_cols, as_index=False).agg(valid_agg)
        # Flatten MultiIndex columns produced when agg contains lists of functions.
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [
                f"{col}_{agg_fn}" if agg_fn else col
                for col, agg_fn in grouped.columns
            ]
        return grouped, (
            f"Grouped {len(df):,} rows by {group_cols} → "
            f"{len(grouped):,} groups | aggregated: {list(valid_agg.keys())}"
        )

    if operation == "rename_columns":
        rename_map: dict = params.get("rename") or {}
        if not rename_map:
            return None, "rename dict required — e.g. {\"old_name\": \"new_name\"}."
        valid = {k: v for k, v in rename_map.items() if k in df.columns}
        if not valid:
            return None, (
                f"None of the columns to rename exist. "
                f"Available: {list(df.columns)[:20]}"
            )
        df = df.rename(columns=valid)
        skipped = len(rename_map) - len(valid)
        detail = f"Renamed {len(valid)} column(s)"
        if skipped:
            detail += f" ({skipped} skipped — not found)"
        return df, detail

    if operation == "dense_panel":
        time_column = params.get("time_column")
        group_by_cols: list = params.get("group_by") or []
        freq = params.get("freq", "M")
        fill_value = params.get("fill_value", 0)
        if not time_column or time_column not in df.columns:
            return None, (
                f"params.time_column required and must exist in dataset. "
                f"Available: {list(df.columns)[:20]}"
            )
        if not group_by_cols:
            return None, "params.group_by must list at least one column."
        missing = [c for c in group_by_cols if c not in df.columns]
        if missing:
            return None, f"group_by columns not found: {missing}"

        # Parse the time column if needed and build the full time index.
        work = df.copy()
        time_series = work[time_column]
        if pd.api.types.is_datetime64_any_dtype(time_series):
            all_times = pd.date_range(time_series.min(), time_series.max(), freq=freq)
        else:
            try:
                parsed = pd.to_datetime(time_series, errors="raise")
                work[time_column] = parsed
                all_times = pd.date_range(parsed.min(), parsed.max(), freq=freq)
            except Exception:
                all_times = sorted(work[time_column].dropna().unique())

        unique_groups = work[group_by_cols].drop_duplicates().reset_index(drop=True)
        times_df = pd.DataFrame({time_column: list(all_times)})
        unique_groups["__xkey"] = 1
        times_df["__xkey"] = 1
        grid = unique_groups.merge(times_df, on="__xkey").drop(columns="__xkey")

        result = grid.merge(work, on=group_by_cols + [time_column], how="left")
        for col in result.columns:
            if col in group_by_cols + [time_column]:
                continue
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(fill_value)
        added = len(result) - len(df)
        return result, (
            f"Densified panel: {len(df):,} → {len(result):,} rows "
            f"(filled {added:,} missing combinations with {fill_value!r})"
        )

    if operation == "create_lag_features":
        column = params.get("column")
        lags: list = params.get("lags") or []
        group_by_cols = params.get("group_by") or []
        time_column = params.get("time_column")
        if not column or column not in df.columns:
            return None, f"params.column required and must exist. Available: {list(df.columns)[:20]}"
        if not lags or not all(isinstance(n, int) and n > 0 for n in lags):
            return None, "params.lags must be a list of positive integers, e.g. [1,3,12]."
        if not group_by_cols:
            return None, "params.group_by required, e.g. [\"shimano_part_no\"]."
        if not time_column or time_column not in df.columns:
            return None, "params.time_column required (used to sort before shifting)."

        result = df.sort_values(group_by_cols + [time_column]).copy()
        added = []
        for lag in lags:
            new_col = f"{column}_lag_{lag}"
            result[new_col] = result.groupby(group_by_cols)[column].shift(lag)
            added.append(new_col)
        return result, f"Added {len(added)} lag feature(s): {added}"

    if operation == "create_rolling_features":
        column = params.get("column")
        windows: list = params.get("windows") or []
        group_by_cols = params.get("group_by") or []
        time_column = params.get("time_column")
        agg = params.get("agg", "mean")
        if not column or column not in df.columns:
            return None, f"params.column required and must exist. Available: {list(df.columns)[:20]}"
        if not windows or not all(isinstance(n, int) and n > 0 for n in windows):
            return None, "params.windows must be a list of positive integers, e.g. [3,6,12]."
        if not group_by_cols:
            return None, "params.group_by required."
        if not time_column or time_column not in df.columns:
            return None, "params.time_column required (used to sort before rolling)."
        if agg not in {"mean", "sum", "std", "min", "max"}:
            return None, "params.agg must be one of mean|sum|std|min|max."

        result = df.sort_values(group_by_cols + [time_column]).copy()
        added = []
        for window in windows:
            new_col = f"{column}_roll_{window}_{agg}"
            # Shift(1) first to avoid leaking the current observation into its own window.
            grouped = result.groupby(group_by_cols)[column]
            shifted = grouped.shift(1)
            result[new_col] = (
                shifted.groupby(result[group_by_cols].apply(tuple, axis=1))
                .transform(lambda s: s.rolling(window=window, min_periods=1).agg(agg))
            )
            added.append(new_col)
        return result, (
            f"Added {len(added)} rolling-{agg} feature(s) (shifted by 1 to prevent leakage): {added}"
        )

    if operation == "create_lead_target":
        column = params.get("column")
        leads: list = params.get("leads") or []
        group_by_cols = params.get("group_by") or []
        time_column = params.get("time_column")
        if not column or column not in df.columns:
            return None, f"params.column required and must exist. Available: {list(df.columns)[:20]}"
        if not leads or not all(isinstance(n, int) and n > 0 for n in leads):
            return None, "params.leads must be a list of positive integers, e.g. [1,3]."
        if not group_by_cols:
            return None, "params.group_by required."
        if not time_column or time_column not in df.columns:
            return None, "params.time_column required (used to sort before shifting forward)."

        result = df.sort_values(group_by_cols + [time_column]).copy()
        added = []
        for lead in leads:
            new_col = f"{column}_lead_{lead}"
            result[new_col] = result.groupby(group_by_cols)[column].shift(-lead)
            added.append(new_col)
        # Drop rows at the tail of each group where the lead target is NaN —
        # those rows have no future observation so they can't be used for
        # training. Keep them if every lead column is NaN handled downstream.
        before = len(result)
        result = result.dropna(subset=added, how="all").reset_index(drop=True)
        dropped = before - len(result)
        return result, (
            f"Added {len(added)} lead target(s): {added} "
            f"(dropped {dropped} tail rows with no future observation)"
        )

    if operation == "execute_python":
        code = (params.get("code") or "").strip()
        if not code:
            return None, "params.code is required — provide Python code that operates on 'df'."
        # Execute in a restricted namespace with only pandas and numpy available.
        # 'df' is exposed as a copy; reassigning df or mutating it in-place both work.
        local_ns: dict = {"df": df.copy(), "pd": pd, "np": np}
        try:
            exec(compile(code, "<custom_fe>", "exec"), local_ns)  # noqa: S102
        except Exception as exc:
            return None, f"Python code execution failed: {exc}"
        result_df = local_ns.get("df")
        if not isinstance(result_df, pd.DataFrame):
            return None, "Code must leave 'df' as a pandas DataFrame."
        if result_df.empty:
            return None, "Code produced an empty DataFrame."
        added = len(result_df.columns) - len(df.columns)
        removed = len(df) - len(result_df)
        detail = (
            f"Custom Python: {len(result_df)} rows × {len(result_df.columns)} cols"
            + (f" (+{added} cols)" if added > 0 else "")
            + (f" (-{abs(added)} cols)" if added < 0 else "")
            + (f" (-{removed} rows)" if removed > 0 else "")
        )
        return result_df, detail

    # ── Extended operations (scaling, encoding, imputation, selection, ──────────
    #    dimensionality reduction, outliers, class-imbalance resampling). These
    #    live in the _NEW_OPERATIONS registry below to keep this dispatcher flat.
    handler = _NEW_OPERATIONS.get(operation)
    if handler is not None:
        return handler(df, params)

    return None, f"Unknown operation: {operation}"


# ──────────────────────────────────────────────────────────────────────────────
# Extended feature-engineering operations
#
# Each helper has the signature (df, params) -> (DataFrame | None, detail_str).
# Returning None signals an error whose message is the detail string. Optional
# third-party libraries (imbalanced-learn, category_encoders) are imported lazily
# so a missing install produces a clear message rather than crashing the run.
# ──────────────────────────────────────────────────────────────────────────────


def _optional_import(module: str, pip_name: str):
    """Import an optional dependency; return (module, None) or (None, error_msg)."""
    try:
        import importlib

        return importlib.import_module(module), None
    except Exception:  # ImportError or partial-install errors
        return None, (
            f"This operation needs the optional package '{pip_name}', which is not "
            f"installed in this environment. Install it with `pip install {pip_name}` "
            f"and try again, or use a different operation."
        )


def _numeric_columns(df: pd.DataFrame, params: dict, *, exclude: list | None = None) -> list[str]:
    """Resolve the numeric columns a transform should act on.

    Uses params.columns when provided (filtered to existing numeric columns),
    otherwise every numeric column. Always drops any names in `exclude`
    (e.g. the target / time / group columns the caller wants left alone).
    """
    exclude = set(exclude or [])
    for key in ("exclude", "exclude_columns"):
        extra = params.get(key)
        if isinstance(extra, list):
            exclude.update(extra)
    requested = params.get("columns")
    if requested:
        cols = [c for c in requested if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        cols = df.select_dtypes(include="number").columns.tolist()
    return [c for c in cols if c not in exclude]


def _coerce_binary_numeric_target(y: pd.Series) -> tuple[pd.Series | None, str]:
    """Map a target to numeric for target-encoding / mean stats.

    Numeric targets pass through. Binary categorical targets are factorised to
    0/1. Multiclass categorical targets are rejected (ambiguous for mean-encoding).
    """
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(float), ""
    nunique = y.nunique(dropna=True)
    if nunique == 2:
        codes, _ = pd.factorize(y)
        return pd.Series(codes, index=y.index).astype(float), ""
    return None, (
        f"target column has {nunique} non-numeric classes — target encoding needs "
        "a numeric (regression) or binary target. One-hot or ordinal encode instead."
    )


# ── Scaling & numeric transforms ────────────────────────────────────────────


def _scale(df: pd.DataFrame, params: dict, scaler, label: str) -> tuple[pd.DataFrame | None, str]:
    cols = _numeric_columns(df, params)
    if not cols:
        return None, "No numeric columns to scale (use params.columns / params.exclude)."
    out = df.copy()
    out[cols] = scaler.fit_transform(out[cols])
    return out, f"{label}: scaled {len(cols)} column(s) {cols[:8]}"


def _op_standard_scale(df, params):
    from sklearn.preprocessing import StandardScaler

    return _scale(df, params, StandardScaler(), "StandardScaler")


def _op_minmax_scale(df, params):
    from sklearn.preprocessing import MinMaxScaler

    return _scale(df, params, MinMaxScaler(), "MinMaxScaler")


def _op_robust_scale(df, params):
    from sklearn.preprocessing import RobustScaler

    return _scale(df, params, RobustScaler(), "RobustScaler")


def _op_max_abs_scale(df, params):
    from sklearn.preprocessing import MaxAbsScaler

    return _scale(df, params, MaxAbsScaler(), "MaxAbsScaler")


def _op_power_transform(df, params):
    from sklearn.preprocessing import PowerTransformer

    method = params.get("method", "yeo-johnson")
    if method not in {"yeo-johnson", "box-cox"}:
        return None, "params.method must be 'yeo-johnson' (default) or 'box-cox'."
    cols = _numeric_columns(df, params)
    if not cols:
        return None, "No numeric columns to transform."
    if method == "box-cox" and (df[cols] <= 0).any().any():
        return None, "box-cox requires strictly positive values; use 'yeo-johnson'."
    out = df.copy()
    out[cols] = PowerTransformer(method=method).fit_transform(out[cols])
    return out, f"PowerTransformer ({method}): transformed {len(cols)} column(s)"


def _op_quantile_transform(df, params):
    from sklearn.preprocessing import QuantileTransformer

    dist = params.get("output_distribution", "normal")
    if dist not in {"normal", "uniform"}:
        return None, "params.output_distribution must be 'normal' (default) or 'uniform'."
    cols = _numeric_columns(df, params)
    if not cols:
        return None, "No numeric columns to transform."
    n_q = min(int(params.get("n_quantiles", 1000)), len(df))
    out = df.copy()
    out[cols] = QuantileTransformer(
        output_distribution=dist, n_quantiles=max(n_q, 2)
    ).fit_transform(out[cols])
    return out, f"QuantileTransformer ({dist}): transformed {len(cols)} column(s)"


def _op_clip_values(df, params):
    col = params.get("column")
    if not col or col not in df.columns:
        return None, f"params.column required and must exist. Available: {list(df.columns)[:20]}"
    lo, hi = params.get("min"), params.get("max")
    if lo is None and hi is None:
        return None, "Provide params.min and/or params.max to clip to."
    out = df.copy()
    out[col] = out[col].clip(lower=lo, upper=hi)
    return out, f"Clipped '{col}' to [{lo}, {hi}]"


def _op_winsorize(df, params):
    lower = float(params.get("lower", 0.01))
    upper = float(params.get("upper", 0.99))
    if not (0 <= lower < upper <= 1):
        return None, "Need 0 <= lower < upper <= 1 (e.g. lower=0.01, upper=0.99)."
    cols = _numeric_columns(df, params)
    if not cols:
        return None, "No numeric columns to winsorize."
    out = df.copy()
    for c in cols:
        lo_v, hi_v = out[c].quantile(lower), out[c].quantile(upper)
        out[c] = out[c].clip(lower=lo_v, upper=hi_v)
    return out, f"Winsorized {len(cols)} column(s) to [{lower:.0%}, {upper:.0%}] percentiles"


# ── Imputation ──────────────────────────────────────────────────────────────


def _op_constant_impute(df, params):
    cols = [c for c in (params.get("columns") or df.columns) if c in df.columns]
    fill_value = params.get("fill_value", 0)
    out = df.copy()
    filled = 0
    for c in cols:
        n = int(out[c].isnull().sum())
        if n:
            out[c] = out[c].fillna(fill_value)
            filled += n
    return out, f"Constant-imputed {filled} missing cell(s) across {len(cols)} column(s) with {fill_value!r}"


def _op_knn_impute(df, params):
    from sklearn.impute import KNNImputer

    cols = _numeric_columns(df, params)
    if not cols:
        return None, "KNN imputation needs numeric columns."
    out = df.copy()
    n_neighbors = max(1, int(params.get("n_neighbors", 5)))
    out[cols] = KNNImputer(n_neighbors=n_neighbors).fit_transform(out[cols])
    return out, f"KNN-imputed {len(cols)} numeric column(s) (n_neighbors={n_neighbors})"


def _op_iterative_impute(df, params):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    cols = _numeric_columns(df, params)
    if not cols:
        return None, "Iterative imputation needs numeric columns."
    out = df.copy()
    max_iter = int(params.get("max_iter", 10))
    out[cols] = IterativeImputer(max_iter=max_iter, random_state=42).fit_transform(out[cols])
    return out, f"Iterative-imputed (MICE) {len(cols)} numeric column(s)"


def _op_add_missing_indicators(df, params):
    cols = params.get("columns") or [c for c in df.columns if df[c].isnull().any()]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return None, "No columns with missing values to flag."
    out = df.copy()
    added = []
    for c in cols:
        flag = f"{c}_was_missing"
        out[flag] = out[c].isnull().astype(int)
        added.append(flag)
    return out, f"Added {len(added)} missing-indicator flag(s): {added[:8]}"


# ── Encoding ──────────────────────────────────────────────────────────────


def _categorical_columns(df: pd.DataFrame, params: dict) -> list[str]:
    requested = params.get("columns")
    if requested:
        return [c for c in requested if c in df.columns]
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def _op_ordinal_encode(df, params):
    from sklearn.preprocessing import OrdinalEncoder

    cols = _categorical_columns(df, params)
    if not cols:
        return None, "No categorical columns to ordinal-encode."
    out = df.copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    out[cols] = enc.fit_transform(out[cols].astype("object").where(out[cols].notnull(), "__nan__"))
    return out, f"Ordinal-encoded {len(cols)} column(s): {cols[:8]}"


def _op_label_encode(df, params):
    col = params.get("column")
    if not col or col not in df.columns:
        return None, f"params.column required and must exist. Available: {list(df.columns)[:20]}"
    out = df.copy()
    codes, _ = pd.factorize(out[col])
    out[col] = codes
    return out, f"Label-encoded '{col}' into integer codes"


def _op_frequency_encode(df, params):
    cols = _categorical_columns(df, params)
    if not cols:
        return None, "No categorical columns to frequency-encode."
    normalize = bool(params.get("normalize", True))
    out = df.copy()
    added = []
    for c in cols:
        freq = out[c].value_counts(normalize=normalize)
        new_col = f"{c}_freq"
        out[new_col] = out[c].map(freq).fillna(0)
        added.append(new_col)
    kind = "frequency" if normalize else "count"
    return out, f"Added {len(added)} {kind}-encoding column(s): {added[:8]}"


def _op_target_encode(df, params):
    target = params.get("target_column")
    if not target or target not in df.columns:
        return None, f"params.target_column required and must exist. Available: {list(df.columns)[:20]}"
    cols = [c for c in (params.get("columns") or _categorical_columns(df, params)) if c in df.columns and c != target]
    if not cols:
        return None, "No categorical columns to target-encode (besides the target)."
    y, err = _coerce_binary_numeric_target(df[target])
    if y is None:
        return None, err
    out = df.copy()
    smoothing = float(params.get("smoothing", 10.0))

    ce, _ = _optional_import("category_encoders", "category-encoders")
    if ce is not None:
        enc = ce.TargetEncoder(cols=cols, smoothing=smoothing)
        out[cols] = enc.fit_transform(out[cols], y)
        return out, f"Target-encoded {len(cols)} column(s) via category_encoders (smoothing={smoothing})"

    # Manual smoothed mean-encoding fallback (global-mean shrinkage).
    global_mean = float(y.mean())
    for c in cols:
        stats = pd.DataFrame({"_y": y.values}, index=df.index).groupby(out[c])["_y"].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        out[c] = out[c].map(smooth).fillna(global_mean)
    return out, f"Target-encoded {len(cols)} column(s) via smoothed mean (smoothing={smoothing})"


def _op_cyclical_encode(df, params):
    col = params.get("column")
    if not col or col not in df.columns:
        return None, f"params.column required and must exist. Available: {list(df.columns)[:20]}"
    if not pd.api.types.is_numeric_dtype(df[col]):
        return None, f"'{col}' must be numeric (e.g. month 1-12, dayofweek 0-6, hour 0-23)."
    period = params.get("period")
    if period is None:
        return None, "params.period required (e.g. 12 for month, 7 for dayofweek, 24 for hour)."
    period = float(period)
    out = df.copy()
    out[f"{col}_sin"] = np.sin(2 * np.pi * out[col] / period)
    out[f"{col}_cos"] = np.cos(2 * np.pi * out[col] / period)
    return out, f"Cyclical-encoded '{col}' (period={period:g}) → {col}_sin, {col}_cos"


def _op_fourier_features(df, params):
    col = params.get("column") or params.get("time_column")
    if not col or col not in df.columns:
        return None, f"params.column (a numeric position or time index) required. Available: {list(df.columns)[:20]}"
    period = params.get("period")
    if period is None:
        return None, "params.period required (the seasonal cycle length, e.g. 12 for monthly-yearly)."
    period = float(period)
    order = max(1, int(params.get("order", 3)))
    series = df[col]
    if not pd.api.types.is_numeric_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().any():
            series = (parsed - parsed.min()).dt.days.astype(float)
        else:
            return None, f"'{col}' is neither numeric nor parseable as a date."
    out = df.copy()
    added = []
    for k in range(1, order + 1):
        s, c = f"fourier_{col}_sin{k}", f"fourier_{col}_cos{k}"
        out[s] = np.sin(2 * np.pi * k * series / period)
        out[c] = np.cos(2 * np.pi * k * series / period)
        added += [s, c]
    return out, f"Added {len(added)} Fourier term(s) for '{col}' (period={period:g}, order={order})"


def _op_datetime_parse(df, params):
    cols = [c for c in (params.get("columns") or []) if c in df.columns]
    if not cols:
        return None, "params.columns required — list the column(s) to parse as datetime."
    out = df.copy()
    parsed = []
    for c in cols:
        converted = pd.to_datetime(out[c], errors="coerce")
        if converted.notna().any():
            out[c] = converted
            parsed.append(c)
    if not parsed:
        return None, f"None of {cols} could be parsed as datetime."
    return out, f"Parsed {len(parsed)} column(s) to datetime: {parsed}"


# ── Cleaning: constant / correlated columns ──────────────────────────────────


def _op_drop_constant(df, params):
    nunique = df.nunique(dropna=False)
    constant = nunique[nunique <= 1].index.tolist()
    if not constant:
        return None, "No constant (zero-variance) columns found."
    return df.drop(columns=constant), f"Dropped {len(constant)} constant column(s): {constant[:12]}"


def _op_drop_correlated(df, params):
    threshold = float(params.get("threshold", 0.95))
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return None, "Need ≥2 numeric columns to assess correlation."
    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    if not to_drop:
        return None, f"No numeric pairs correlated above {threshold}."
    return df.drop(columns=to_drop), f"Dropped {len(to_drop)} column(s) correlated > {threshold}: {to_drop[:12]}"


# ── Feature selection & dimensionality reduction ─────────────────────────────


def _select_setup(df, params):
    """Shared validation for supervised selectors. Returns (X_num, y, task, target, err)."""
    target = params.get("target_column")
    if not target or target not in df.columns:
        return None, None, None, None, (
            f"params.target_column required and must exist. Available: {list(df.columns)[:20]}"
        )
    from ..training import infer_task_type

    y = df[target]
    task = infer_task_type(y)
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    if len(num_cols) < 2:
        return None, None, None, None, "Need ≥2 numeric feature columns for selection."
    X = df[num_cols].fillna(df[num_cols].median())
    return X, y, task, target, ""


def _op_select_k_best(df, params):
    from sklearn.feature_selection import (
        SelectKBest,
        f_classif,
        f_regression,
        mutual_info_classif,
        mutual_info_regression,
    )

    X, y, task, target, err = _select_setup(df, params)
    if X is None:
        return None, err
    k = int(params.get("k", min(10, X.shape[1])))
    k = max(1, min(k, X.shape[1]))
    use_mi = bool(params.get("mutual_info", False))
    if task == "classification":
        score_func = mutual_info_classif if use_mi else f_classif
    else:
        score_func = mutual_info_regression if use_mi else f_regression
    selector = SelectKBest(score_func=score_func, k=k).fit(X, y)
    keep = X.columns[selector.get_support()].tolist()
    dropped_num = [c for c in X.columns if c not in keep]
    out = df.drop(columns=dropped_num)
    metric = "mutual information" if use_mi else "F-test"
    return out, f"SelectKBest ({metric}, {task}): kept top {k} numeric features {keep}, dropped {len(dropped_num)}"


def _op_select_from_model(df, params):
    from sklearn.feature_selection import SelectFromModel

    X, y, task, target, err = _select_setup(df, params)
    if X is None:
        return None, err
    if task == "classification":
        from sklearn.ensemble import RandomForestClassifier

        est = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        from sklearn.ensemble import RandomForestRegressor

        est = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    threshold = params.get("threshold", "median")
    selector = SelectFromModel(est, threshold=threshold).fit(X, y)
    keep = X.columns[selector.get_support()].tolist()
    if not keep:
        return None, "SelectFromModel kept no features — lower the threshold."
    dropped_num = [c for c in X.columns if c not in keep]
    return df.drop(columns=dropped_num), (
        f"SelectFromModel (RandomForest importance ≥ {threshold!r}): kept {len(keep)} {keep}, dropped {len(dropped_num)}"
    )


def _op_rfe_select(df, params):
    from sklearn.feature_selection import RFE

    X, y, task, target, err = _select_setup(df, params)
    if X is None:
        return None, err
    n = int(params.get("n_features_to_select", min(10, X.shape[1])))
    n = max(1, min(n, X.shape[1]))
    if task == "classification":
        from sklearn.ensemble import RandomForestClassifier

        est = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
    else:
        from sklearn.ensemble import RandomForestRegressor

        est = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
    selector = RFE(est, n_features_to_select=n).fit(X, y)
    keep = X.columns[selector.get_support()].tolist()
    dropped_num = [c for c in X.columns if c not in keep]
    return df.drop(columns=dropped_num), f"RFE: kept {n} feature(s) {keep}, dropped {len(dropped_num)}"


def _op_pca(df, params):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    target = params.get("target_column")
    keep_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) or c == target]
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    if len(num_cols) < 2:
        return None, "Need ≥2 numeric columns for PCA."
    n_components = params.get("n_components", min(5, len(num_cols)))
    if isinstance(n_components, (int, float)) and n_components >= 1:
        n_components = min(int(n_components), len(num_cols))
    X = StandardScaler().fit_transform(df[num_cols].fillna(df[num_cols].median()))
    pca = PCA(n_components=n_components, random_state=42)
    comps = pca.fit_transform(X)
    comp_df = pd.DataFrame(
        comps, columns=[f"pc_{i+1}" for i in range(comps.shape[1])], index=df.index
    )
    out = pd.concat([df[keep_cols].reset_index(drop=True), comp_df.reset_index(drop=True)], axis=1)
    evr = pca.explained_variance_ratio_.sum()
    return out, (
        f"PCA: reduced {len(num_cols)} numeric features → {comps.shape[1]} components "
        f"({evr:.1%} variance retained); kept {keep_cols}"
    )


# ── Outlier handling ─────────────────────────────────────────────────────────


def _op_zscore_outlier_removal(df, params):
    threshold = float(params.get("threshold", 3.0))
    cols = _numeric_columns(df, params, exclude=[params.get("target_column")] if params.get("target_column") else None)
    if not cols:
        return None, "No numeric columns for z-score outlier removal."
    before = len(df)
    sub = df[cols]
    z = (sub - sub.mean()) / sub.std(ddof=0).replace(0, np.nan)
    mask = (z.abs() <= threshold) | z.isna()
    out = df[mask.all(axis=1)].reset_index(drop=True)
    return out, f"Removed {before - len(out)} row(s) with |z| > {threshold} in {len(cols)} column(s)"


def _op_isolation_forest_outliers(df, params):
    from sklearn.ensemble import IsolationForest

    cols = _numeric_columns(df, params, exclude=[params.get("target_column")] if params.get("target_column") else None)
    if not cols:
        return None, "Isolation Forest needs numeric columns."
    contamination = params.get("contamination", "auto")
    X = df[cols].fillna(df[cols].median())
    preds = IsolationForest(contamination=contamination, random_state=42).fit_predict(X)
    before = len(df)
    out = df[preds == 1].reset_index(drop=True)
    return out, f"Isolation Forest removed {before - len(out)} anomalous row(s) (contamination={contamination})"


# ── Class-imbalance resampling (imbalanced-learn) ────────────────────────────


def _resample(df, params, make_sampler, label, *, numeric_only=False):
    target = params.get("target_column")
    if not target or target not in df.columns:
        return None, f"params.target_column required and must exist. Available: {list(df.columns)[:20]}"
    imblearn, err = _optional_import("imblearn", "imbalanced-learn")
    if imblearn is None:
        return None, err

    y = df[target]
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 20:
        return None, "Resampling is for classification targets; this target looks continuous."
    X = df.drop(columns=[target])
    if X.isnull().any().any():
        return None, "Resamplers cannot handle missing values — impute first (e.g. impute_missing)."

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if numeric_only and cat_cols:
        return None, (
            f"{label} only supports numeric features; encode categoricals first "
            f"(categorical columns: {cat_cols[:8]}) or use operation 'smote' (auto-SMOTENC) / 'random_oversample'."
        )

    counts = y.value_counts()
    min_count = int(counts.min())
    k_neighbors = max(1, min(int(params.get("k_neighbors", 5)), min_count - 1)) if min_count > 1 else 1

    try:
        sampler = make_sampler(imblearn, X, cat_cols, k_neighbors)
        X_res, y_res = sampler.fit_resample(X, y)
    except Exception as exc:
        return None, f"{label} failed: {exc}"

    out = X_res.copy()
    out[target] = y_res
    out = out[df.columns]
    new_counts = pd.Series(y_res).value_counts().to_dict()
    return out, (
        f"{label}: {len(df):,} → {len(out):,} rows. "
        f"Class balance now {new_counts} (was {counts.to_dict()})"
    )


def _op_smote(df, params):
    def make(imblearn, X, cat_cols, k):
        if cat_cols:
            from imblearn.over_sampling import SMOTENC

            cat_idx = [X.columns.get_loc(c) for c in cat_cols]
            if len(cat_idx) == X.shape[1]:
                from imblearn.over_sampling import RandomOverSampler

                return RandomOverSampler(random_state=42)
            return SMOTENC(categorical_features=cat_idx, random_state=42, k_neighbors=k)
        from imblearn.over_sampling import SMOTE

        return SMOTE(random_state=42, k_neighbors=k)

    return _resample(df, params, make, "SMOTE")


def _op_borderline_smote(df, params):
    def make(imblearn, X, cat_cols, k):
        from imblearn.over_sampling import BorderlineSMOTE

        return BorderlineSMOTE(random_state=42, k_neighbors=k)

    return _resample(df, params, make, "BorderlineSMOTE", numeric_only=True)


def _op_adasyn(df, params):
    def make(imblearn, X, cat_cols, k):
        from imblearn.over_sampling import ADASYN

        return ADASYN(random_state=42, n_neighbors=k)

    return _resample(df, params, make, "ADASYN", numeric_only=True)


def _op_random_oversample(df, params):
    def make(imblearn, X, cat_cols, k):
        from imblearn.over_sampling import RandomOverSampler

        return RandomOverSampler(random_state=42)

    return _resample(df, params, make, "RandomOverSampler")


def _op_random_undersample(df, params):
    def make(imblearn, X, cat_cols, k):
        from imblearn.under_sampling import RandomUnderSampler

        return RandomUnderSampler(random_state=42)

    return _resample(df, params, make, "RandomUnderSampler")


def _op_smote_tomek(df, params):
    def make(imblearn, X, cat_cols, k):
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE

        return SMOTETomek(random_state=42, smote=SMOTE(random_state=42, k_neighbors=k))

    return _resample(df, params, make, "SMOTETomek", numeric_only=True)


def _op_smote_enn(df, params):
    def make(imblearn, X, cat_cols, k):
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE

        return SMOTEENN(random_state=42, smote=SMOTE(random_state=42, k_neighbors=k))

    return _resample(df, params, make, "SMOTEENN", numeric_only=True)


def _op_hash_encode(df, params):
    from sklearn.feature_extraction import FeatureHasher

    cols = _categorical_columns(df, params)
    if not cols:
        return None, "No categorical columns to hash-encode."
    n_features = int(params.get("n_features", 64))
    out = df.copy()
    added = []
    for c in cols:
        hasher = FeatureHasher(n_features=n_features, input_type="string")
        hashed = hasher.transform(out[c].astype(str).apply(lambda x: [x]))
        hashed_df = pd.DataFrame(
            hashed.toarray(),
            columns=[f"{c}_hash_{i}" for i in range(n_features)],
            index=out.index,
        )
        out = pd.concat([out.drop(columns=[c]), hashed_df], axis=1)
        added.extend(hashed_df.columns.tolist())
    return out, f"Hash-encoded {len(cols)} column(s) → {len(added)} features (n_features={n_features})"


# ── Registry: operation name → handler ───────────────────────────────────────

_NEW_OPERATIONS: dict[str, Any] = {
    # scaling / numeric transforms
    "standard_scale": _op_standard_scale,
    "minmax_scale": _op_minmax_scale,
    "robust_scale": _op_robust_scale,
    "max_abs_scale": _op_max_abs_scale,
    "power_transform": _op_power_transform,
    "quantile_transform": _op_quantile_transform,
    "clip_values": _op_clip_values,
    "winsorize": _op_winsorize,
    # imputation
    "constant_impute": _op_constant_impute,
    "knn_impute": _op_knn_impute,
    "iterative_impute": _op_iterative_impute,
    "add_missing_indicators": _op_add_missing_indicators,
    # encoding
    "ordinal_encode": _op_ordinal_encode,
    "label_encode": _op_label_encode,
    "frequency_encode": _op_frequency_encode,
    "target_encode": _op_target_encode,
    "hash_encode": _op_hash_encode,
    "cyclical_encode": _op_cyclical_encode,
    "fourier_features": _op_fourier_features,
    "datetime_parse": _op_datetime_parse,
    # cleaning
    "drop_constant": _op_drop_constant,
    "drop_correlated": _op_drop_correlated,
    # selection / reduction
    "select_k_best": _op_select_k_best,
    "select_from_model": _op_select_from_model,
    "rfe_select": _op_rfe_select,
    "pca": _op_pca,
    # outliers
    "zscore_outlier_removal": _op_zscore_outlier_removal,
    "isolation_forest_outliers": _op_isolation_forest_outliers,
    # class-imbalance resampling
    "smote": _op_smote,
    "borderline_smote": _op_borderline_smote,
    "adasyn": _op_adasyn,
    "random_oversample": _op_random_oversample,
    "random_undersample": _op_random_undersample,
    "smote_tomek": _op_smote_tomek,
    "smote_enn": _op_smote_enn,
}
