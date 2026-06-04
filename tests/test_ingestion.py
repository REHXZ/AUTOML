from __future__ import annotations

import sqlite3

import pandas as pd

from backend.logic.ingestion import load_dataset, list_sqlite_tables


def test_load_csv(tmp_path):
    path = tmp_path / "customers.csv"
    pd.DataFrame({"age": [31, 42], "churn": [0, 1]}).to_csv(path, index=False)

    loaded = load_dataset(path)

    assert loaded.source_type == "CSV"
    assert loaded.dataframe.shape == (2, 2)
    assert loaded.dataframe["churn"].tolist() == [0, 1]


def test_load_excel(tmp_path):
    path = tmp_path / "customers.xlsx"
    pd.DataFrame({"segment": ["A", "B"], "revenue": [100, 125]}).to_excel(path, index=False)

    loaded = load_dataset(path)

    assert loaded.source_type == "Excel"
    assert loaded.dataframe.shape == (2, 2)


def test_load_json_records(tmp_path):
    path = tmp_path / "customers.json"
    pd.DataFrame({"segment": ["A", "B"], "churn": [0, 1]}).to_json(path, orient="records")

    loaded = load_dataset(path)

    assert loaded.source_type == "JSON"
    assert loaded.dataframe["segment"].tolist() == ["A", "B"]


def test_load_sqlite_table(tmp_path):
    path = tmp_path / "warehouse.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE customers (age INTEGER, churn INTEGER)")
        connection.executemany("INSERT INTO customers VALUES (?, ?)", [(31, 0), (42, 1)])

    loaded = load_dataset(path, table_name="customers")

    assert list_sqlite_tables(path) == ["customers"]
    assert loaded.source_type == "SQLite"
    assert loaded.table_name == "customers"
    assert loaded.dataframe.shape == (2, 2)

