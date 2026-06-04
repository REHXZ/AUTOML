from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend.config.settings import SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    path: Path
    dataframe: pd.DataFrame
    source_type: str
    table_name: str | None = None


def validate_source_path(path: str | Path) -> Path:
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported types: {supported}.")
    if not source_path.exists():
        raise FileNotFoundError(f"Data source not found: {source_path}")
    return source_path


def list_sqlite_tables(path: str | Path) -> list[str]:
    source_path = validate_source_path(path)
    with sqlite3.connect(source_path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [row[0] for row in rows]


def load_dataset(path: str | Path, table_name: str | None = None) -> LoadedDataset:
    source_path = validate_source_path(path)
    suffix = source_path.suffix.lower()

    if suffix == ".csv":
        dataframe = pd.read_csv(source_path)
        source_type = "CSV"
        dataset_name = source_path.stem
    elif suffix in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(source_path)
        source_type = "Excel"
        dataset_name = source_path.stem
    elif suffix == ".json":
        dataframe = _read_json(source_path)
        source_type = "JSON"
        dataset_name = source_path.stem
    elif suffix in {".db", ".sqlite", ".sqlite3"}:
        table = table_name or _first_sqlite_table(source_path)
        dataframe = _read_sqlite_table(source_path, table)
        source_type = "SQLite"
        dataset_name = f"{source_path.stem}.{table}"
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if dataframe.empty:
        raise ValueError(f"{dataset_name} did not contain any rows.")

    return LoadedDataset(
        name=dataset_name,
        path=source_path,
        dataframe=dataframe,
        source_type=source_type,
        table_name=table if source_type == "SQLite" else None,
    )


def _read_json(path: Path) -> pd.DataFrame:
    try:
        dataframe = pd.read_json(path)
    except ValueError:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        dataframe = pd.json_normalize(payload)

    if isinstance(dataframe, pd.Series):
        dataframe = dataframe.to_frame()
    return dataframe


def _first_sqlite_table(path: Path) -> str:
    tables = list_sqlite_tables(path)
    if not tables:
        raise ValueError(f"No tables found in SQLite source: {path}")
    return tables[0]


def _read_sqlite_table(path: Path, table_name: str) -> pd.DataFrame:
    tables = list_sqlite_tables(path)
    if table_name not in tables:
        available = ", ".join(tables) or "none"
        raise ValueError(f"Table '{table_name}' not found. Available tables: {available}.")

    escaped_table = table_name.replace('"', '""')
    with sqlite3.connect(path) as connection:
        return pd.read_sql_query(f'SELECT * FROM "{escaped_table}"', connection)
