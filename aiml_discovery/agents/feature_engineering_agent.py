"""Feature Engineering Agent: creates derived datasets via transformations."""

from __future__ import annotations

import json
import logging
from typing import Any, Generator

import numpy as np
import pandas as pd

from ..ingestion import load_dataset
from ..logging_setup import configure_logging
from ..storage import DatasetInfo
from .base import AgentContext, AutopilotStep, BaseAgent, to_json_safe

configure_logging()
log = logging.getLogger(__name__)

# Per-operation parameter keys the LLM sometimes sends at the top level of the
# tool args instead of nested under "params". We auto-promote them to params
# (with a WARNING) so the operation still runs.
_FE_PARAM_KEYS = frozenset({
    "threshold",          # drop_high_missing
    "columns",            # select/drop_columns, one_hot_encode, log_transform, polynomial_features
    "max_unique",         # one_hot_encode
    "column",             # bin_numeric, target_log_transform, lag/lead/rolling
    "bins",               # bin_numeric
    "pairs",              # interaction_features
    "degree",             # polynomial_features
    "group_by",           # groupby_aggregate, lag/lead/rolling, dense_panel
    "aggregations",       # groupby_aggregate
    "rename",             # rename_columns
    "time_column",        # lag/lead/rolling, dense_panel
    "lags",               # create_lag_features
    "leads",              # create_lead_target
    "windows",            # create_rolling_features
    "agg",                # create_rolling_features
    "freq",               # dense_panel
    "fill_value",         # dense_panel
    "code",               # execute_python
})
# Operations that require params at all — used to escalate the missing-params warning.
_OPS_REQUIRING_PARAMS = frozenset({
    "select_columns", "drop_columns", "one_hot_encode", "log_transform",
    "bin_numeric", "interaction_features", "polynomial_features",
    "target_log_transform", "groupby_aggregate", "rename_columns",
    "dense_panel", "create_lag_features", "create_rolling_features",
    "create_lead_target", "execute_python",
})


_SYSTEM_PROMPT = """\
You are the Feature Engineering Agent — you transform raw datasets into
modelling-ready ones to maximise downstream signal.

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
Basic cleaning:
  • drop_high_missing   — drop cols with > params.threshold missing (default 0.5)
  • drop_duplicates     — remove duplicate rows
  • select_columns      — keep only params.columns
  • drop_columns        — remove params.columns
  • filter_outliers     — IQR outlier removal on numeric cols (1.5 IQR)
  • impute_missing      — fill numeric with median, categorical with mode

Encoding & scaling:
  • encode_dates        — expand datetime → year/month/day/dayofweek/quarter
  • one_hot_encode      — one-hot params.columns (or auto low-cardinality)
  • log_transform       — log1p params.columns
  • bin_numeric         — quantile-bin params.column into params.bins
  • interaction_features— multiplicative params.pairs: [[a,b], ...]
  • polynomial_features — squared/cubed of params.columns (params.degree)
  • target_log_transform— log1p of params.column (regression skew fix)
  • rename_columns      — params.rename: {"old":"new", ...}

Aggregation:
  • groupby_aggregate   — params.group_by + params.aggregations
                          Aggs: count|nunique|sum|mean|min|max|first|last
                          USE for monthly/weekly rollups of transactional rows.

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

    return None, f"Unknown operation: {operation}"
