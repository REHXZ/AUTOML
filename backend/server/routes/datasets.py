from pathlib import Path

from flask import Blueprint, abort, jsonify, request

from backend.logic.ingestion import list_sqlite_tables, load_dataset
from backend.server.auth import get_current_user_id
from backend.server.helpers import project_or_404
from backend.services.project_store import DatasetInfo, ProjectStore

datasets_bp = Blueprint("datasets", __name__)


@datasets_bp.get("/api/projects/<project_id>/datasets")
def list_datasets_api(project_id: str):
    user_id = get_current_user_id()
    store = ProjectStore()
    project_or_404(store, project_id, user_id=user_id)
    datasets = store.list_datasets(project_id)
    return jsonify({"project_id": project_id, "datasets": [d.to_dict() for d in datasets]})


@datasets_bp.post("/api/projects/<project_id>/datasets/upload")
def upload_dataset_api(project_id: str):
    user_id = get_current_user_id()
    store = ProjectStore()
    project_or_404(store, project_id, user_id=user_id)

    file = request.files.get("file")
    if file is None:
        abort(400, description="Dataset file is required.")
    filename = Path(file.filename or "").name
    if not filename:
        abort(400, description="Dataset filename is required.")
    data = file.read()
    if not data:
        abort(400, description="Dataset file is empty.")

    name = (request.form.get("name") or "").strip() or None
    table_name = (request.form.get("table_name") or "").strip() or None

    saved_path = store.save_dataset_file(project_id, filename, data)
    try:
        datasets = _register_dataset_path(
            store=store,
            project_id=project_id,
            path=saved_path,
            name=name,
            source_name=filename,
            table_name=table_name,
        )
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        abort(400, description=str(exc))

    return jsonify({"project_id": project_id, "datasets": [d.to_dict() for d in datasets]}), 201


@datasets_bp.post("/api/projects/<project_id>/datasets/register")
def register_dataset_api(project_id: str):
    user_id = get_current_user_id()
    store = ProjectStore()
    project_or_404(store, project_id, user_id=user_id)

    body = request.get_json(silent=True) or {}
    file_path_str = (body.get("file_path") or "").strip()
    if not file_path_str:
        abort(400, description="file_path is required.")

    path = Path(file_path_str).expanduser()
    if not path.exists() or not path.is_file():
        abort(400, description=f"Dataset file not found: {path}")

    datasets = _register_dataset_path(
        store=store,
        project_id=project_id,
        path=path,
        name=body.get("name"),
        source_name=body.get("source_name"),
        table_name=body.get("table_name"),
    )
    return jsonify({"project_id": project_id, "datasets": [d.to_dict() for d in datasets]}), 201


def _register_dataset_path(
    *,
    store: ProjectStore,
    project_id: str,
    path: Path,
    name: str | None,
    source_name: str | None,
    table_name: str | None,
) -> list[DatasetInfo]:
    suffix = path.suffix.lower()
    registered: list[DatasetInfo] = []

    if suffix in {".db", ".sqlite", ".sqlite3"} and table_name is None:
        table_names = list_sqlite_tables(path)
        if not table_names:
            abort(400, description="No tables were found in the SQLite source.")
    else:
        table_names = [table_name]

    for table in table_names:
        try:
            loaded = load_dataset(path, table_name=table)
        except Exception as exc:
            abort(400, description=str(exc))

        dataset_name = name or loaded.name
        if table is not None and name is None:
            dataset_name = f"{path.stem}.{table}"

        registered.append(
            store.register_dataset(
                project_id,
                name=dataset_name,
                source_name=source_name or path.name,
                source_type=loaded.source_type,
                file_path=path,
                table_name=table,
                row_count=len(loaded.dataframe),
                column_count=len(loaded.dataframe.columns),
            )
        )

    return registered
