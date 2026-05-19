from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from aiml_discovery.config import PROJECT_HOME


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
    def __init__(self, root: str | Path = PROJECT_HOME) -> None:
        self.root = Path(root).expanduser()

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> list[ProjectInfo]:
        self.ensure_root()
        projects = []
        for project_path in sorted(self.root.iterdir()):
            if not project_path.is_dir():
                continue
            metadata_path = project_path / "project.json"
            if not metadata_path.exists():
                continue
            metadata = _read_json(metadata_path)
            projects.append(_project_from_metadata(metadata, project_path))
        return sorted(projects, key=lambda project: project.updated_at, reverse=True)

    def create_project(self, name: str, description: str = "") -> ProjectInfo:
        self.ensure_root()
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("Project name is required.")

        project_id = self._unique_project_id(cleaned_name)
        project_path = self.root / project_id
        for folder in ["datasets", "runs", "reports"]:
            (project_path / folder).mkdir(parents=True, exist_ok=True)

        now = _utc_now()
        metadata = {
            "id": project_id,
            "name": cleaned_name,
            "description": description.strip(),
            "created_at": now,
            "updated_at": now,
        }
        _write_json(project_path / "project.json", metadata)
        _write_json(project_path / "datasets.json", [])
        return _project_from_metadata(metadata, project_path)

    def get_project(self, project_id: str) -> ProjectInfo:
        project_path = self.project_path(project_id)
        metadata = _read_json(project_path / "project.json")
        return _project_from_metadata(metadata, project_path)

    def project_path(self, project_id: str) -> Path:
        project_path = self.root / project_id
        if not project_path.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        return project_path

    def save_dataset_file(self, project_id: str, filename: str, data: bytes) -> Path:
        project_path = self.project_path(project_id)
        datasets_path = project_path / "datasets"
        datasets_path.mkdir(parents=True, exist_ok=True)

        original = Path(filename)
        safe_stem = _slugify(original.stem) or "dataset"
        suffix = original.suffix.lower()
        destination = datasets_path / f"{safe_stem}{suffix}"
        counter = 2
        while destination.exists():
            destination = datasets_path / f"{safe_stem}_{counter}{suffix}"
            counter += 1

        destination.write_bytes(data)
        self._touch_project(project_id)
        return destination

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
        dataset = DatasetInfo(
            id=f"dataset_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}",
            name=name,
            source_name=source_name,
            source_type=source_type,
            file_path=str(file_path),
            table_name=table_name,
            row_count=int(row_count),
            column_count=int(column_count),
            uploaded_at=_utc_now(),
        )
        datasets = [dataset.to_dict(), *[item.to_dict() for item in self.list_datasets(project_id)]]
        _write_json(self.project_path(project_id) / "datasets.json", datasets)
        self._touch_project(project_id)
        return dataset

    def list_datasets(self, project_id: str) -> list[DatasetInfo]:
        catalog_path = self.project_path(project_id) / "datasets.json"
        if not catalog_path.exists():
            return []
        datasets = _read_json(catalog_path)
        return [DatasetInfo(**item) for item in datasets]

    def save_run(
        self,
        project_id: str,
        run_metadata: Mapping[str, Any],
        model: Any,
        report_text: str,
    ) -> Path:
        import joblib

        project_path = self.project_path(project_id)
        run_id = str(run_metadata.get("run_id") or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S"))
        run_path = project_path / "runs" / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        model_path = run_path / "model.joblib"
        report_path = run_path / "report.md"
        metadata_path = run_path / "metadata.json"

        joblib.dump(model, model_path)
        report_path.write_text(report_text, encoding="utf-8")

        metadata = _jsonable(
            {
                **dict(run_metadata),
                "model_path": str(model_path),
                "report_path": str(report_path),
                "saved_at": _utc_now(),
            }
        )
        _write_json(metadata_path, metadata)
        self._touch_project(project_id)
        return run_path

    def list_runs(self, project_id: str) -> list[dict[str, Any]]:
        runs_path = self.project_path(project_id) / "runs"
        if not runs_path.exists():
            return []

        runs = []
        for run_path in runs_path.iterdir():
            metadata_path = run_path / "metadata.json"
            if metadata_path.exists():
                runs.append(_read_json(metadata_path))
        return sorted(runs, key=lambda run: run.get("trained_at", ""), reverse=True)

    def read_report(self, report_path: str | Path) -> str:
        return Path(report_path).read_text(encoding="utf-8")

    def _unique_project_id(self, name: str) -> str:
        base = _slugify(name) or "project"
        project_id = base
        counter = 2
        while (self.root / project_id).exists():
            project_id = f"{base}_{counter}"
            counter += 1
        return project_id

    def _touch_project(self, project_id: str) -> None:
        project_path = self.project_path(project_id)
        metadata_path = project_path / "project.json"
        metadata = _read_json(metadata_path)
        metadata["updated_at"] = _utc_now()
        _write_json(metadata_path, metadata)


def _project_from_metadata(metadata: Mapping[str, Any], path: Path) -> ProjectInfo:
    return ProjectInfo(
        id=str(metadata["id"]),
        name=str(metadata["name"]),
        description=str(metadata.get("description", "")),
        created_at=str(metadata["created_at"]),
        updated_at=str(metadata["updated_at"]),
        path=path,
    )


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value

