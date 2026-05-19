from __future__ import annotations

from typing import Any

import pandas as pd


def profile_dataframe(dataframe: pd.DataFrame) -> dict[str, Any]:
    row_count = int(len(dataframe))
    column_count = int(len(dataframe.columns))
    duplicate_rows = int(dataframe.duplicated().sum()) if row_count else 0
    memory_mb = float(dataframe.memory_usage(deep=True).sum() / 1_000_000)

    column_profiles = [_profile_column(dataframe[column], row_count) for column in dataframe.columns]
    missing_cells = int(dataframe.isna().sum().sum())
    total_cells = row_count * column_count

    return {
        "row_count": row_count,
        "column_count": column_count,
        "duplicate_rows": duplicate_rows,
        "missing_cells": missing_cells,
        "missing_pct": float((missing_cells / total_cells) * 100) if total_cells else 0.0,
        "memory_mb": memory_mb,
        "columns": column_profiles,
        "numeric_summary": _numeric_summary(dataframe),
    }


def _profile_column(series: pd.Series, row_count: int) -> dict[str, Any]:
    missing_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))
    sample_values = [str(value) for value in series.dropna().head(5).tolist()]

    return {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "role": _infer_column_role(series),
        "missing_count": missing_count,
        "missing_pct": float((missing_count / row_count) * 100) if row_count else 0.0,
        "unique_count": unique_count,
        "sample_values": sample_values,
    }


def _infer_column_role(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "flag"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    unique_count = series.nunique(dropna=True)
    row_count = len(series)
    if row_count and unique_count / row_count > 0.8:
        return "identifier/text"
    return "categorical"


def _numeric_summary(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    numeric = dataframe.select_dtypes(include="number")
    if numeric.empty:
        return []

    summary = numeric.describe().transpose().reset_index(names="column")
    return summary.round(4).to_dict(orient="records")

