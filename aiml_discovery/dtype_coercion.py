"""Automatic dtype coercion for DataFrames loaded from flat files.

Applies two passes:
  1. object columns → datetime64 when the column name suggests a timestamp
     OR when >90 % of sampled values parse successfully as dates.
  2. object columns → numeric when almost all non-null values are numeric strings.

Returns the coerced DataFrame plus a change-log dict so callers can report
what was changed to the LLM.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

log = logging.getLogger(__name__)

# Regex that matches column names strongly suggesting a datetime value.
_DATETIME_NAME_RE = re.compile(
    r"(^|[_\s\-\.])"
    r"(date|time|timestamp|datetime|dt|"
    r"created|updated|modified|"
    r"start|end|opened|closed|"
    r"birth|expir|"
    r"arrival|departure|scheduled|"
    r"posted|submitted|recorded|logged|at)"
    r"([_\s\-\.]|$)",
    re.IGNORECASE,
)

# If a name hint is present, accept this fraction of parseable values.
_THRESHOLD_WITH_NAME = 0.80
# Without a name hint we demand a very high parse rate to avoid false positives.
_THRESHOLD_NO_NAME = 0.95
# Extra NaTs introduced by coercion must stay below this fraction.
_MAX_NEW_NULL_RATE = 0.05
# Numeric coercion: tolerate at most this fraction of new NaNs.
_MAX_NUMERIC_NULL_RATE = 0.02


def coerce_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return (coerced_df, changes) where changes maps col_name → description."""
    df = df.copy()
    changes: dict[str, str] = {}

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            continue
        if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
            continue

        coerced = _try_datetime(series, col)
        if coerced is not None:
            df[col] = coerced
            changes[col] = f"object → datetime64[ns]"
            log.info("dtype_coercion | %r coerced to datetime64", col)
            continue

        coerced = _try_numeric(series, col)
        if coerced is not None:
            df[col] = coerced
            changes[col] = f"object → {coerced.dtype}"
            log.info("dtype_coercion | %r coerced to %s", col, coerced.dtype)

    return df, changes


# ── helpers ───────────────────────────────────────────────────────────────────

def _name_suggests_datetime(col: str) -> bool:
    return bool(_DATETIME_NAME_RE.search(col))


def _try_datetime(series: pd.Series, col: str) -> pd.Series | None:
    non_null = series.dropna()
    if non_null.empty:
        return None

    name_hint = _name_suggests_datetime(col)
    threshold = _THRESHOLD_WITH_NAME if name_hint else _THRESHOLD_NO_NAME

    sample = non_null.head(200)
    try:
        parsed_sample = pd.to_datetime(sample, errors="coerce")
    except Exception:
        return None

    if parsed_sample.notna().mean() < threshold:
        return None

    try:
        result = pd.to_datetime(series, errors="coerce")
    except Exception:
        return None

    original_nulls = int(series.isna().sum())
    new_nulls = int(result.isna().sum())
    extra_null_rate = (new_nulls - original_nulls) / max(len(series), 1)
    if extra_null_rate > _MAX_NEW_NULL_RATE:
        return None

    return result


def _try_numeric(series: pd.Series, col: str) -> pd.Series | None:
    non_null = series.dropna()
    if non_null.empty:
        return None

    sample = non_null.head(100)
    if pd.to_numeric(sample, errors="coerce").isna().mean() > _MAX_NUMERIC_NULL_RATE:
        return None

    result = pd.to_numeric(series, errors="coerce")
    original_nulls = int(series.isna().sum())
    new_nulls = int(result.isna().sum())
    extra_null_rate = (new_nulls - original_nulls) / max(len(series), 1)
    if extra_null_rate > _MAX_NUMERIC_NULL_RATE:
        return None

    return result
