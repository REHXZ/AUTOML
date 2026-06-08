from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.config.settings import FILE_CACHE_DIR
from backend.services.db import get_backend


@dataclass(frozen=True)
class ProjectInfo:
    id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    path: Path


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    name: str
    source_name: str
    source_type: str
    file_path: str
    table_name: str | None
    row_count: int
    column_count: int
    uploaded_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProjectStore:
    """Thin façade over the active StorageBackend. Public API is unchanged."""

    def __init__(self, root=None) -> None:
        # `root` kept for call-site compatibility; storage location is controlled by env vars.
        self._backend = get_backend()

    def ensure_root(self) -> None:
        FILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Projects ──────────────────────────────────────────────────────────────

    def list_projects(self, user_id: str = "local") -> list[ProjectInfo]:
        return [_project_from_row(r) for r in self._backend.list_projects(user_id)]

    def create_project(self, name: str, description: str = "", user_id: str = "local") -> ProjectInfo:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("Project name is required.")
        project_id = self._unique_project_id(cleaned, user_id)
        now = _utc_now()
        self._backend.create_project(project_id, cleaned, description.strip(), now, now, user_id)
        return ProjectInfo(
            id=project_id, name=cleaned, description=description.strip(),
            created_at=now, updated_at=now,
            path=_project_cache_dir(project_id),
        )

    def get_project(self, project_id: str, user_id: str | None = None) -> ProjectInfo:
        row = self._backend.get_project(project_id, user_id)
        if row is None:
            raise FileNotFoundError(f"Project not found: {project_id}")
        return _project_from_row(row)

    def project_path(self, project_id: str) -> Path:
        """Return (and create) the local cache directory for this project."""
        path = _project_cache_dir(project_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── Datasets ──────────────────────────────────────────────────────────────

    def save_dataset_file(self, project_id: str, filename: str, data: bytes) -> Path:
        """Write the raw upload bytes to the project cache dir and return the path."""
        original = Path(filename)
        safe_stem = _slugify(original.stem) or "dataset"
        suffix = original.suffix.lower()
        upload_dir = _project_cache_dir(project_id) / "datasets"
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / f"{safe_stem}{suffix}"
        counter = 2
        while dest.exists():
            dest = upload_dir / f"{safe_stem}_{counter}{suffix}"
            counter += 1
        dest.write_bytes(data)
        self._backend.touch_project(project_id, _utc_now())
        return dest

    def register_dataset(
        self,
        project_id: str,
        *,
        name: str,
        source_name: str,
        source_type: str,
        file_path: str | Path,
        row_count: int,
        column_count: int,
        table_name: str | None = None,
    ) -> DatasetInfo:
        dataset_id = f"dataset_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        # Write bytes to the canonical per-dataset cache so file_path is stable.
        file_bytes = Path(file_path).read_bytes()
        canonical = FILE_CACHE_DIR / "datasets" / dataset_id / source_name
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_bytes(file_bytes)

        dataset = DatasetInfo(
            id=dataset_id,
            name=name,
            source_name=source_name,
            source_type=source_type,
            file_path=str(canonical),
            table_name=table_name,
            row_count=int(row_count),
            column_count=int(column_count),
            uploaded_at=_utc_now(),
        )
        self._backend.insert_dataset(project_id, dataset.to_dict(), file_bytes)
        self._backend.touch_project(project_id, _utc_now())
        return dataset

    def list_datasets(self, project_id: str) -> list[DatasetInfo]:
        rows = self._backend.list_datasets(project_id)
        result: list[DatasetInfo] = []
        for row in rows:
            source_name = row.get("source_name", "")
            dataset_id = row["id"]
            # Resolve to a real local path, downloading from backend if the cache is cold.
            try:
                file_path = str(self._backend.resolve_dataset_path(dataset_id, source_name))
            except Exception:
                file_path = ""
            result.append(DatasetInfo(
                id=dataset_id,
                name=row["name"],
                source_name=source_name,
                source_type=row.get("source_type", ""),
                file_path=file_path,
                table_name=row.get("table_name"),
                row_count=int(row.get("row_count") or 0),
                column_count=int(row.get("column_count") or 0),
                uploaded_at=row.get("uploaded_at", ""),
            ))
        return result

    # ── Training Runs ─────────────────────────────────────────────────────────

    def save_run(
        self,
        project_id: str,
        run_metadata: Mapping[str, Any],
        model: Any,
        report_text: str,
    ) -> Path:
        import joblib

        run_id = str(
            run_metadata.get("run_id")
            or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        )
        session_id = str(run_metadata.get("session_id") or "") or None

        # Serialize model to bytes.
        buf = io.BytesIO()
        joblib.dump(model, buf)
        model_bytes = buf.getvalue()

        # Write model and report to local cache so routes can access them by path.
        run_cache = FILE_CACHE_DIR / "runs" / run_id
        run_cache.mkdir(parents=True, exist_ok=True)
        model_path = run_cache / "model.joblib"
        report_path = run_cache / "report.md"
        model_path.write_bytes(model_bytes)
        report_path.write_text(report_text, encoding="utf-8")

        now = _utc_now()
        metadata = _jsonable({
            **dict(run_metadata),
            "run_id": run_id,
            "model_path": str(model_path),
            "report_path": str(report_path),
            "saved_at": now,
        })

        self._backend.insert_run(project_id, run_id, session_id, metadata, model_bytes, report_text)
        self._backend.touch_project(project_id, now)
        return run_cache

    def list_runs(self, project_id: str) -> list[dict[str, Any]]:
        return self._backend.list_runs(project_id)

    def read_report(self, report_path: str | Path) -> str:
        path = Path(str(report_path))
        if path.exists():
            return path.read_text(encoding="utf-8")
        # Fall back to looking up the report in the DB by treating the stem as run_id.
        run_id = path.stem
        report = self._backend.get_run_report(run_id)
        if report is not None:
            return report
        raise FileNotFoundError(f"Report not found: {report_path}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _unique_project_id(self, name: str, user_id: str = "local") -> str:
        base = _slugify(name) or "project"
        project_id = base
        counter = 2
        # Check uniqueness globally (project IDs are shared across users in the same backend).
        while self._backend.get_project(project_id) is not None:
            project_id = f"{base}_{counter}"
            counter += 1
        return project_id


# ── Module-level helpers ───────────────────────────────────────────────────────


def _project_cache_dir(project_id: str) -> Path:
    return FILE_CACHE_DIR / "projects" / project_id


def _project_from_row(row: dict) -> ProjectInfo:
    pid = str(row["id"])
    return ProjectInfo(
        id=pid,
        name=str(row["name"]),
        description=str(row.get("description", "")),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        path=_project_cache_dir(pid),
    )


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value
