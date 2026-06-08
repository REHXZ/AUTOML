from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .base import StorageBackend

_SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'local'
);

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    source_name TEXT DEFAULT '',
    source_type TEXT DEFAULT '',
    table_name TEXT,
    row_count INTEGER DEFAULT 0,
    column_count INTEGER DEFAULT 0,
    uploaded_at TEXT NOT NULL,
    file_bytes BLOB
);

CREATE TABLE IF NOT EXISTS training_runs (
    run_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    session_id TEXT,
    model_bytes BLOB,
    report_md TEXT DEFAULT '',
    metadata_json TEXT DEFAULT '{}',
    trained_at TEXT,
    saved_at TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_goal TEXT DEFAULT '',
    title TEXT DEFAULT '',
    status TEXT DEFAULT 'running',
    strategy_summary TEXT DEFAULT '',
    step_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    step_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_notebook (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    notebook_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS session_new_datasets (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    datasets_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS session_training_runs (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    runs_json TEXT NOT NULL DEFAULT '[]'
);
"""


class SQLiteBackend(StorageBackend):
    """Local SQLite storage. Thread-safe via per-thread connections in WAL mode."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Bootstrap schema on a dedicated connection, then close it.
        boot = sqlite3.connect(str(self._path))
        boot.executescript(_SCHEMA)
        # Migrate existing DBs that pre-date the user_id column.
        try:
            boot.execute("ALTER TABLE projects ADD COLUMN user_id TEXT NOT NULL DEFAULT 'local'")
            boot.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
        boot.close()

    @property
    def _con(self) -> sqlite3.Connection:
        if not hasattr(self._local, "con"):
            con = sqlite3.connect(str(self._path), check_same_thread=False)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            con.execute("PRAGMA journal_mode = WAL")
            self._local.con = con
        return self._local.con

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        cur = self._con.execute(sql, params)
        self._con.commit()
        return cur

    def _row(self, sql: str, params: tuple = ()) -> dict | None:
        row = self._con.execute(sql, params).fetchone()
        return dict(row) if row else None

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        return [dict(r) for r in self._con.execute(sql, params).fetchall()]

    # ── Projects ──────────────────────────────────────────────────────────────

    def list_projects(self, user_id: str = "local") -> list[dict]:
        return self._rows(
            "SELECT id,name,description,created_at,updated_at FROM projects "
            "WHERE user_id=? ORDER BY updated_at DESC",
            (user_id,),
        )

    def create_project(self, id, name, description, created_at, updated_at, user_id: str = "local") -> None:
        self._exec(
            "INSERT INTO projects(id,name,description,created_at,updated_at,user_id) VALUES(?,?,?,?,?,?)",
            (id, name, description, created_at, updated_at, user_id),
        )

    def get_project(self, project_id, user_id: str | None = None) -> dict | None:
        if user_id is not None:
            return self._row(
                "SELECT id,name,description,created_at,updated_at FROM projects WHERE id=? AND user_id=?",
                (project_id, user_id),
            )
        return self._row(
            "SELECT id,name,description,created_at,updated_at FROM projects WHERE id=?",
            (project_id,),
        )

    def touch_project(self, project_id, updated_at) -> None:
        self._exec("UPDATE projects SET updated_at=? WHERE id=?", (updated_at, project_id))

    # ── Datasets ──────────────────────────────────────────────────────────────

    def list_datasets(self, project_id) -> list[dict]:
        return self._rows(
            "SELECT id,project_id,name,source_name,source_type,table_name,"
            "row_count,column_count,uploaded_at FROM datasets WHERE project_id=? ORDER BY uploaded_at DESC",
            (project_id,),
        )

    def insert_dataset(self, project_id, dataset: dict, file_bytes: bytes) -> None:
        self._exec(
            "INSERT OR REPLACE INTO datasets"
            "(id,project_id,name,source_name,source_type,table_name,"
            "row_count,column_count,uploaded_at,file_bytes) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                dataset["id"], project_id, dataset["name"],
                dataset.get("source_name", ""), dataset.get("source_type", ""),
                dataset.get("table_name"),
                dataset.get("row_count", 0), dataset.get("column_count", 0),
                dataset["uploaded_at"], file_bytes,
            ),
        )

    def get_dataset_bytes(self, dataset_id) -> bytes | None:
        row = self._row("SELECT file_bytes FROM datasets WHERE id=?", (dataset_id,))
        return bytes(row["file_bytes"]) if row and row["file_bytes"] is not None else None

    # ── Training Runs ─────────────────────────────────────────────────────────

    def list_runs(self, project_id) -> list[dict]:
        rows = self._rows(
            "SELECT run_id,project_id,session_id,metadata_json,report_md,trained_at,saved_at "
            "FROM training_runs WHERE project_id=? ORDER BY trained_at DESC",
            (project_id,),
        )
        for row in rows:
            meta = json.loads(row.pop("metadata_json") or "{}")
            row.update(meta)
        return rows

    def insert_run(self, project_id, run_id, session_id, metadata, model_bytes, report_md) -> None:
        self._exec(
            "INSERT OR REPLACE INTO training_runs"
            "(run_id,project_id,session_id,model_bytes,report_md,metadata_json,trained_at,saved_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (
                run_id, project_id, session_id, model_bytes, report_md,
                json.dumps(metadata),
                metadata.get("trained_at", ""), metadata.get("saved_at", ""),
            ),
        )

    def get_run_model_bytes(self, run_id) -> bytes | None:
        row = self._row("SELECT model_bytes FROM training_runs WHERE run_id=?", (run_id,))
        return bytes(row["model_bytes"]) if row and row["model_bytes"] is not None else None

    def get_run_report(self, run_id) -> str | None:
        row = self._row("SELECT report_md FROM training_runs WHERE run_id=?", (run_id,))
        return row["report_md"] if row else None

    def get_run_metadata(self, run_id) -> dict | None:
        row = self._row(
            "SELECT run_id,project_id,session_id,metadata_json,trained_at,saved_at "
            "FROM training_runs WHERE run_id=?",
            (run_id,),
        )
        if row is None:
            return None
        meta = json.loads(row.pop("metadata_json") or "{}")
        return {**row, **meta}

    # ── Sessions ──────────────────────────────────────────────────────────────

    def list_sessions(self, project_id) -> list[dict]:
        return self._rows(
            "SELECT id,project_id,user_goal,title,status,strategy_summary,"
            "step_count,created_at,updated_at FROM sessions WHERE project_id=? ORDER BY updated_at DESC",
            (project_id,),
        )

    def upsert_session(self, project_id, session_id, **fields) -> None:
        existing = self._row("SELECT id FROM sessions WHERE id=?", (session_id,))
        if existing is None:
            defaults: dict[str, Any] = {
                "id": session_id, "project_id": project_id,
                "user_goal": "", "title": "", "status": "running",
                "strategy_summary": "", "step_count": 0,
                "created_at": "", "updated_at": "",
            }
            defaults.update(fields)
            self._exec(
                "INSERT INTO sessions(id,project_id,user_goal,title,status,"
                "strategy_summary,step_count,created_at,updated_at) "
                "VALUES(:id,:project_id,:user_goal,:title,:status,"
                ":strategy_summary,:step_count,:created_at,:updated_at)",
                defaults,
            )
        elif fields:
            sets = ", ".join(f"{k}=?" for k in fields)
            self._exec(
                f"UPDATE sessions SET {sets} WHERE id=?",
                (*fields.values(), session_id),
            )

    def get_session(self, project_id, session_id) -> dict | None:
        return self._row(
            "SELECT id,project_id,user_goal,title,status,strategy_summary,"
            "step_count,created_at,updated_at FROM sessions WHERE id=? AND project_id=?",
            (session_id, project_id),
        )

    def delete_session(self, project_id, session_id) -> None:
        self._exec(
            "DELETE FROM sessions WHERE id=? AND project_id=?", (session_id, project_id)
        )

    # ── Session content ───────────────────────────────────────────────────────

    def append_step(self, session_id, step_json) -> None:
        self._exec(
            "INSERT INTO session_steps(session_id,step_json) VALUES(?,?)",
            (session_id, step_json),
        )

    def get_steps(self, session_id) -> list[str]:
        return [
            r["step_json"]
            for r in self._rows(
                "SELECT step_json FROM session_steps WHERE session_id=? ORDER BY id",
                (session_id,),
            )
        ]

    def append_message(self, session_id, message_json) -> None:
        self._exec(
            "INSERT INTO session_messages(session_id,message_json) VALUES(?,?)",
            (session_id, message_json),
        )

    def get_messages(self, session_id) -> list[str]:
        return [
            r["message_json"]
            for r in self._rows(
                "SELECT message_json FROM session_messages WHERE session_id=? ORDER BY id",
                (session_id,),
            )
        ]

    def save_notebook(self, session_id, notebook_json) -> None:
        self._exec(
            "INSERT OR REPLACE INTO session_notebook(session_id,notebook_json) VALUES(?,?)",
            (session_id, notebook_json),
        )

    def get_notebook(self, session_id) -> str:
        row = self._row(
            "SELECT notebook_json FROM session_notebook WHERE session_id=?", (session_id,)
        )
        return row["notebook_json"] if row else "[]"

    def save_session_datasets(self, session_id, datasets_json) -> None:
        self._exec(
            "INSERT OR REPLACE INTO session_new_datasets(session_id,datasets_json) VALUES(?,?)",
            (session_id, datasets_json),
        )

    def get_session_datasets(self, session_id) -> str:
        row = self._row(
            "SELECT datasets_json FROM session_new_datasets WHERE session_id=?", (session_id,)
        )
        return row["datasets_json"] if row else "[]"

    def save_session_runs(self, session_id, runs_json) -> None:
        self._exec(
            "INSERT OR REPLACE INTO session_training_runs(session_id,runs_json) VALUES(?,?)",
            (session_id, runs_json),
        )

    def get_session_runs(self, session_id) -> str:
        row = self._row(
            "SELECT runs_json FROM session_training_runs WHERE session_id=?", (session_id,)
        )
        return row["runs_json"] if row else "[]"
