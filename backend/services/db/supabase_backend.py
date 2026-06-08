from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .base import StorageBackend

log = logging.getLogger(__name__)

_BUCKET = "automl-files"


class SupabaseBackend(StorageBackend):
    """Supabase-backed storage. Metadata in PostgreSQL, files in Supabase Storage."""

    def __init__(self, url: str, key: str) -> None:
        from supabase import create_client
        self._client = create_client(url, key)
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self._client.storage.create_bucket(_BUCKET, options={"public": False})
        except Exception:
            pass  # bucket already exists

    def _t(self, table: str):
        return self._client.table(table)

    # ── Projects ──────────────────────────────────────────────────────────────

    def list_projects(self, user_id: str = "local") -> list[dict]:
        return (
            self._t("projects")
            .select("id,name,description,created_at,updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
            .data or []
        )

    def create_project(self, id, name, description, created_at, updated_at, user_id: str = "local") -> None:
        self._t("projects").insert({
            "id": id, "name": name, "description": description,
            "created_at": created_at, "updated_at": updated_at, "user_id": user_id,
        }).execute()

    def get_project(self, project_id, user_id: str | None = None) -> dict | None:
        q = (
            self._t("projects")
            .select("id,name,description,created_at,updated_at")
            .eq("id", project_id)
        )
        if user_id is not None:
            q = q.eq("user_id", user_id)
        rows = q.execute().data
        return rows[0] if rows else None

    def touch_project(self, project_id, updated_at) -> None:
        self._t("projects").update({"updated_at": updated_at}).eq("id", project_id).execute()

    # ── Datasets ──────────────────────────────────────────────────────────────

    def list_datasets(self, project_id) -> list[dict]:
        return (
            self._t("datasets")
            .select("id,project_id,name,source_name,source_type,table_name,row_count,column_count,uploaded_at")
            .eq("project_id", project_id)
            .order("uploaded_at", desc=True)
            .execute()
            .data or []
        )

    def insert_dataset(self, project_id, dataset: dict, file_bytes: bytes) -> None:
        source_name = dataset.get("source_name") or dataset["id"]
        storage_path = f"datasets/{dataset['id']}/{source_name}"
        self._client.storage.from_(_BUCKET).upload(
            storage_path,
            file_bytes,
            file_options={"content-type": "application/octet-stream"},
        )
        self._t("datasets").insert({
            "id": dataset["id"],
            "project_id": project_id,
            "name": dataset["name"],
            "source_name": dataset.get("source_name", ""),
            "source_type": dataset.get("source_type", ""),
            "storage_path": storage_path,
            "table_name": dataset.get("table_name"),
            "row_count": dataset.get("row_count", 0),
            "column_count": dataset.get("column_count", 0),
            "uploaded_at": dataset["uploaded_at"],
        }).execute()

    def get_dataset_bytes(self, dataset_id) -> bytes | None:
        rows = self._t("datasets").select("storage_path").eq("id", dataset_id).execute().data
        if not rows or not rows[0].get("storage_path"):
            return None
        try:
            return self._client.storage.from_(_BUCKET).download(rows[0]["storage_path"])
        except Exception as exc:
            log.warning("SupabaseBackend.get_dataset_bytes | download failed: %s", exc)
            return None

    # ── Training Runs ─────────────────────────────────────────────────────────

    def list_runs(self, project_id) -> list[dict]:
        rows = (
            self._t("training_runs")
            .select("run_id,project_id,session_id,metadata_json,report_md,trained_at,saved_at")
            .eq("project_id", project_id)
            .order("trained_at", desc=True)
            .execute()
            .data or []
        )
        for row in rows:
            meta = json.loads(row.pop("metadata_json") or "{}")
            row.update(meta)
        return rows

    def insert_run(self, project_id, run_id, session_id, metadata, model_bytes, report_md) -> None:
        storage_path = f"models/{run_id}/model.joblib"
        try:
            self._client.storage.from_(_BUCKET).upload(
                storage_path,
                model_bytes,
                file_options={"content-type": "application/octet-stream"},
            )
        except Exception as exc:
            log.warning("SupabaseBackend.insert_run | model upload failed: %s", exc)
            storage_path = None
        self._t("training_runs").insert({
            "run_id": run_id,
            "project_id": project_id,
            "session_id": session_id,
            "storage_path": storage_path,
            "report_md": report_md,
            "metadata_json": json.dumps(metadata),
            "trained_at": metadata.get("trained_at", ""),
            "saved_at": metadata.get("saved_at", ""),
        }).execute()

    def get_run_model_bytes(self, run_id) -> bytes | None:
        rows = (
            self._t("training_runs").select("storage_path").eq("run_id", run_id).execute().data
        )
        if not rows or not rows[0].get("storage_path"):
            return None
        try:
            return self._client.storage.from_(_BUCKET).download(rows[0]["storage_path"])
        except Exception as exc:
            log.warning("SupabaseBackend.get_run_model_bytes | download failed: %s", exc)
            return None

    def get_run_report(self, run_id) -> str | None:
        rows = self._t("training_runs").select("report_md").eq("run_id", run_id).execute().data
        return rows[0]["report_md"] if rows else None

    def get_run_metadata(self, run_id) -> dict | None:
        rows = (
            self._t("training_runs")
            .select("run_id,project_id,session_id,metadata_json,trained_at,saved_at")
            .eq("run_id", run_id)
            .execute()
            .data
        )
        if not rows:
            return None
        row = dict(rows[0])
        meta = json.loads(row.pop("metadata_json") or "{}")
        return {**row, **meta}

    # ── Sessions ──────────────────────────────────────────────────────────────

    def list_sessions(self, project_id) -> list[dict]:
        return (
            self._t("sessions")
            .select("id,project_id,user_goal,title,status,strategy_summary,step_count,created_at,updated_at")
            .eq("project_id", project_id)
            .order("updated_at", desc=True)
            .execute()
            .data or []
        )

    def upsert_session(self, project_id, session_id, **fields) -> None:
        row: dict[str, Any] = {"id": session_id, "project_id": project_id}
        row.update(fields)
        self._t("sessions").upsert(row).execute()

    def get_session(self, project_id, session_id) -> dict | None:
        rows = (
            self._t("sessions")
            .select("id,project_id,user_goal,title,status,strategy_summary,step_count,created_at,updated_at")
            .eq("id", session_id)
            .eq("project_id", project_id)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def delete_session(self, project_id, session_id) -> None:
        self._t("sessions").delete().eq("id", session_id).eq("project_id", project_id).execute()

    # ── Session content ───────────────────────────────────────────────────────

    def append_step(self, session_id, step_json) -> None:
        self._t("session_steps").insert({"session_id": session_id, "step_json": step_json}).execute()

    def get_steps(self, session_id) -> list[str]:
        rows = (
            self._t("session_steps")
            .select("step_json")
            .eq("session_id", session_id)
            .order("id")
            .execute()
            .data or []
        )
        return [r["step_json"] for r in rows]

    def append_message(self, session_id, message_json) -> None:
        self._t("session_messages").insert(
            {"session_id": session_id, "message_json": message_json}
        ).execute()

    def get_messages(self, session_id) -> list[str]:
        rows = (
            self._t("session_messages")
            .select("message_json")
            .eq("session_id", session_id)
            .order("id")
            .execute()
            .data or []
        )
        return [r["message_json"] for r in rows]

    def save_notebook(self, session_id, notebook_json) -> None:
        self._t("session_notebook").upsert(
            {"session_id": session_id, "notebook_json": notebook_json}
        ).execute()

    def get_notebook(self, session_id) -> str:
        rows = (
            self._t("session_notebook")
            .select("notebook_json")
            .eq("session_id", session_id)
            .execute()
            .data
        )
        return rows[0]["notebook_json"] if rows else "[]"

    def save_session_datasets(self, session_id, datasets_json) -> None:
        self._t("session_new_datasets").upsert(
            {"session_id": session_id, "datasets_json": datasets_json}
        ).execute()

    def get_session_datasets(self, session_id) -> str:
        rows = (
            self._t("session_new_datasets")
            .select("datasets_json")
            .eq("session_id", session_id)
            .execute()
            .data
        )
        return rows[0]["datasets_json"] if rows else "[]"

    def save_session_runs(self, session_id, runs_json) -> None:
        self._t("session_training_runs").upsert(
            {"session_id": session_id, "runs_json": runs_json}
        ).execute()

    def get_session_runs(self, session_id) -> str:
        rows = (
            self._t("session_training_runs")
            .select("runs_json")
            .eq("session_id", session_id)
            .execute()
            .data
        )
        return rows[0]["runs_json"] if rows else "[]"
