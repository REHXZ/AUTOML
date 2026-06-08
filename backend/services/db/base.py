from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class StorageBackend(ABC):
    """Abstract storage backend. SQLiteBackend for local dev, SupabaseBackend for hosted."""

    # ── Projects ──────────────────────────────────────────────────────────────

    @abstractmethod
    def list_projects(self, user_id: str = "local") -> list[dict[str, Any]]: ...

    @abstractmethod
    def create_project(
        self, id: str, name: str, description: str, created_at: str, updated_at: str,
        user_id: str = "local",
    ) -> None: ...

    @abstractmethod
    def get_project(self, project_id: str, user_id: str | None = None) -> dict[str, Any] | None: ...

    @abstractmethod
    def touch_project(self, project_id: str, updated_at: str) -> None: ...

    # ── Datasets ──────────────────────────────────────────────────────────────

    @abstractmethod
    def list_datasets(self, project_id: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def insert_dataset(
        self, project_id: str, dataset: dict[str, Any], file_bytes: bytes
    ) -> None: ...

    @abstractmethod
    def get_dataset_bytes(self, dataset_id: str) -> bytes | None: ...

    # ── Training Runs ─────────────────────────────────────────────────────────

    @abstractmethod
    def list_runs(self, project_id: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def insert_run(
        self,
        project_id: str,
        run_id: str,
        session_id: str | None,
        metadata: dict[str, Any],
        model_bytes: bytes,
        report_md: str,
    ) -> None: ...

    @abstractmethod
    def get_run_model_bytes(self, run_id: str) -> bytes | None: ...

    @abstractmethod
    def get_run_report(self, run_id: str) -> str | None: ...

    @abstractmethod
    def get_run_metadata(self, run_id: str) -> dict[str, Any] | None: ...

    # ── Sessions ──────────────────────────────────────────────────────────────

    @abstractmethod
    def list_sessions(self, project_id: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def upsert_session(
        self, project_id: str, session_id: str, **fields: Any
    ) -> None: ...

    @abstractmethod
    def get_session(
        self, project_id: str, session_id: str
    ) -> dict[str, Any] | None: ...

    @abstractmethod
    def delete_session(self, project_id: str, session_id: str) -> None: ...

    # ── Session content ───────────────────────────────────────────────────────

    @abstractmethod
    def append_step(self, session_id: str, step_json: str) -> None: ...

    @abstractmethod
    def get_steps(self, session_id: str) -> list[str]: ...

    @abstractmethod
    def append_message(self, session_id: str, message_json: str) -> None: ...

    @abstractmethod
    def get_messages(self, session_id: str) -> list[str]: ...

    @abstractmethod
    def save_notebook(self, session_id: str, notebook_json: str) -> None: ...

    @abstractmethod
    def get_notebook(self, session_id: str) -> str: ...

    @abstractmethod
    def save_session_datasets(self, session_id: str, datasets_json: str) -> None: ...

    @abstractmethod
    def get_session_datasets(self, session_id: str) -> str: ...

    @abstractmethod
    def save_session_runs(self, session_id: str, runs_json: str) -> None: ...

    @abstractmethod
    def get_session_runs(self, session_id: str) -> str: ...

    # ── File resolution ───────────────────────────────────────────────────────

    def resolve_dataset_path(self, dataset_id: str, source_name: str) -> Path:
        """Return a local path for this dataset, extracting from storage if the cache is cold."""
        cache = _cache_dir() / "datasets" / dataset_id
        target = cache / source_name
        if not target.exists():
            data = self.get_dataset_bytes(dataset_id)
            if data is None:
                raise FileNotFoundError(f"Dataset {dataset_id!r} not found in storage")
            cache.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return target

    def resolve_run_model_path(self, run_id: str) -> Path:
        """Return a local path for this run's model, extracting from storage if needed."""
        cache = _cache_dir() / "runs" / run_id
        target = cache / "model.joblib"
        if not target.exists():
            data = self.get_run_model_bytes(run_id)
            if data is None:
                raise FileNotFoundError(f"Model for run {run_id!r} not found in storage")
            cache.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return target


def _cache_dir() -> Path:
    from backend.config.settings import FILE_CACHE_DIR
    return FILE_CACHE_DIR
