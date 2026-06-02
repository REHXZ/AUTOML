"""HTTP API for AIML Discovery and the AI Autopilot.

Run locally with:

    python server.py          # uses HOST/PORT env vars, default 0.0.0.0:8082

The API keeps the existing AiAutopilot class as the single execution engine.
Long-running autopilot sessions run in a background thread and persist their
steps through the same session store used by the Streamlit UI.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Generator

from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, request, stream_with_context
from werkzeug.exceptions import HTTPException

from aiml_discovery.ai_autopilot import AiAutopilot, AutopilotStep
from aiml_discovery.config import APP_NAME
from aiml_discovery.ingestion import load_dataset, list_sqlite_tables
from aiml_discovery.logging_setup import configure_logging
from aiml_discovery.notebook_export import build_notebook, serialize_notebook
from aiml_discovery.session_store import (
    SessionWriter,
    delete_session,
    list_sessions,
    load_session,
    step_to_jsonable,
)
from aiml_discovery.storage import DatasetInfo, ProjectInfo, ProjectStore
from aiml_discovery.tracing import configure_tracing

load_dotenv()
configure_logging()
configure_tracing()

HOST = os.environ.get("API_HOST", "0.0.0.0")
PORT = int(os.environ.get("API_PORT", 8082))

STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_WAITING = "waiting_for_input"
STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"
TERMINAL_STATUSES = {STATUS_IDLE, STATUS_COMPLETE, STATUS_ERROR}


@dataclass
class AutopilotJob:
    project_id: str
    session_id: str
    autopilot: AiAutopilot
    generator: Generator[AutopilotStep, list[str] | None, None] | None = None
    status: str = STATUS_IDLE
    pending_step: dict[str, Any] | None = None
    error: str | None = None
    worker: Thread | None = None
    lock: Lock = field(default_factory=Lock)
    should_stop: bool = False


app = Flask(__name__)

_jobs: dict[tuple[str, str], AutopilotJob] = {}
_jobs_lock = Lock()


@app.errorhandler(HTTPException)
def _http_error(exc: HTTPException):
    return jsonify({"detail": exc.description}), exc.code


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "aiml-discovery-api",
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
    })


@app.get("/api/projects")
def list_projects_api():
    store = ProjectStore()
    return jsonify({"projects": [_project_payload(p) for p in store.list_projects()]})


@app.post("/api/projects")
def create_project_api():
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        abort(400, description="name is required and must be non-empty.")
    store = ProjectStore()
    try:
        project = store.create_project(name, body.get("description", ""))
    except ValueError as exc:
        abort(400, description=str(exc))
    return jsonify({"project": _project_payload(project)}), 201


@app.get("/api/projects/<project_id>")
def get_project_api(project_id: str):
    store = ProjectStore()
    project = _project_or_404(store, project_id)
    return jsonify({"project": _project_payload(project)})


@app.get("/api/projects/<project_id>/datasets")
def list_datasets_api(project_id: str):
    store = ProjectStore()
    _project_or_404(store, project_id)
    datasets = store.list_datasets(project_id)
    return jsonify({"project_id": project_id, "datasets": [d.to_dict() for d in datasets]})


@app.post("/api/projects/<project_id>/datasets/upload")
def upload_dataset_api(project_id: str):
    store = ProjectStore()
    _project_or_404(store, project_id)

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


@app.post("/api/projects/<project_id>/datasets/register")
def register_dataset_api(project_id: str):
    store = ProjectStore()
    _project_or_404(store, project_id)

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


@app.get("/api/projects/<project_id>/autopilot/sessions")
def list_autopilot_sessions_api(project_id: str):
    store = ProjectStore()
    _project_or_404(store, project_id)
    return jsonify({
        "project_id": project_id,
        "sessions": [asdict(r) for r in list_sessions(store, project_id)],
    })


@app.post("/api/projects/<project_id>/autopilot/sessions")
def start_autopilot_session_api(project_id: str):
    store = ProjectStore()
    _project_or_404(store, project_id)
    _require_datasets(store, project_id)

    body = request.get_json(silent=True) or {}
    api_key = _resolve_api_key(body.get("api_key"))
    autopilot = AiAutopilot(
        api_key=api_key,
        project_id=project_id,
        store=store,
        user_goal=(body.get("user_goal") or "").strip(),
    )
    job = AutopilotJob(
        project_id=project_id,
        session_id=autopilot.session_id,
        autopilot=autopilot,
        generator=autopilot.run(),
    )
    _register_job(job)
    _launch_job(job)
    return jsonify(_job_response(job)), 202


@app.get("/api/projects/<project_id>/autopilot/sessions/<session_id>")
def get_autopilot_session_api(project_id: str, session_id: str):
    store = ProjectStore()
    _project_or_404(store, project_id)
    return jsonify(_session_payload(store, project_id, session_id))


@app.delete("/api/projects/<project_id>/autopilot/sessions/<session_id>")
def delete_autopilot_session_api(project_id: str, session_id: str):
    store = ProjectStore()
    _project_or_404(store, project_id)

    job = _get_job(project_id, session_id)
    if job is not None and job.worker is not None and job.worker.is_alive():
        abort(409, description="Cannot delete an autopilot session while it is running.")

    delete_session(store, project_id, session_id)
    _remove_job(project_id, session_id)
    return Response("", status=204)


@app.get("/api/projects/<project_id>/autopilot/sessions/<session_id>/events")
def stream_autopilot_events_api(project_id: str, session_id: str):
    store = ProjectStore()
    _project_or_404(store, project_id)
    _load_session_or_404(store, project_id, session_id)

    from_index = max(0, request.args.get("from_index", 0, type=int))
    return Response(
        stream_with_context(
            _event_stream(str(store.root), project_id, session_id, from_index)
        ),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/projects/<project_id>/autopilot/sessions/<session_id>/answers")
def answer_autopilot_questions_api(project_id: str, session_id: str):
    store = ProjectStore()
    _project_or_404(store, project_id)
    _load_session_or_404(store, project_id, session_id)

    job = _get_job(project_id, session_id)
    if job is None or job.generator is None:
        abort(409, description=(
            "This session is not waiting inside the current API process. "
            "Restart the run or continue from a completed session."
        ))
    if job.status != STATUS_WAITING:
        abort(409, description=f"Session is not waiting for answers; current status is {job.status}.")

    body = request.get_json(silent=True) or {}
    answers = body.get("answers", [])
    _launch_job(job, answers=answers)
    return jsonify(_job_response(job)), 202


@app.post("/api/projects/<project_id>/autopilot/sessions/<session_id>/messages")
def continue_autopilot_session_api(project_id: str, session_id: str):
    store = ProjectStore()
    _project_or_404(store, project_id)
    loaded = _load_session_or_404(store, project_id, session_id)

    job = _get_job(project_id, session_id)
    if job is not None and job.worker is not None and job.worker.is_alive():
        abort(409, description="Session is already running.")
    if job is not None and job.status == STATUS_WAITING:
        abort(409, description="Session is waiting for answers. Submit answers before sending a follow-up.")
    if job is None and loaded.status == STATUS_WAITING:
        abort(409, description="Session is waiting for answers. Submit answers before sending a follow-up.")
    if job is None and loaded.status == STATUS_RUNNING:
        _recover_orphaned_session(store, project_id, session_id, loaded)

    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        abort(400, description="message is required and must be non-empty.")

    api_key = _resolve_api_key(body.get("api_key"))
    if job is None:
        autopilot = AiAutopilot(
            api_key=api_key,
            project_id=project_id,
            store=store,
            session_id=session_id,
            preloaded_session=loaded,
        )
        job = AutopilotJob(
            project_id=project_id,
            session_id=session_id,
            autopilot=autopilot,
        )
        _register_job(job)

    job.generator = job.autopilot.continue_with(message)
    _launch_job(job)
    return jsonify(_job_response(job)), 202


@app.post("/api/projects/<project_id>/autopilot/sessions/<session_id>/stop")
def stop_autopilot_session_api(project_id: str, session_id: str):
    store = ProjectStore()
    _project_or_404(store, project_id)
    job = _get_job(project_id, session_id)

    if job is not None and job.worker is not None and job.worker.is_alive():
        job.autopilot.signal_stop()
        with job.lock:
            job.should_stop = True
        return jsonify(_job_response(job)), 202

    loaded = _load_session_or_404(store, project_id, session_id)
    if loaded.status in {STATUS_RUNNING, STATUS_WAITING}:
        _recover_orphaned_session(store, project_id, session_id, loaded)
        return jsonify({
            "project_id": project_id,
            "session_id": session_id,
            "status": STATUS_IDLE,
            "pending_step": None,
            "error": None,
            "worker_alive": False,
        }), 202

    abort(409, description=f"Session is not running (current status: {loaded.status}).")


@app.get("/api/projects/<project_id>/runs")
def list_runs_api(project_id: str):
    store = ProjectStore()
    _project_or_404(store, project_id)
    runs = store.list_runs(project_id)
    safe = [
        {k: v for k, v in r.items() if k not in {"diagnostics", "leaderboard"}}
        for r in runs
    ]
    return jsonify({"runs": safe})


@app.post("/api/projects/<project_id>/runs/<run_id>/score")
def score_run_api(project_id: str, run_id: str):
    """Score new rows using a previously saved model pipeline.

    Accepts either:
      - JSON body: {"data": [{col: val, ...}, ...]}
      - multipart/form-data with a "file" CSV field

    Returns: {"run_id": ..., "task_type": ..., "predictions": [...]}
    """
    import io
    import joblib
    import pandas as pd

    store = ProjectStore()
    _project_or_404(store, project_id)

    runs = store.list_runs(project_id)
    run = next((r for r in runs if r.get("run_id") == run_id), None)
    if run is None:
        abort(404, description=f"Run '{run_id}' not found in project '{project_id}'.")

    model_path = run.get("model_path")
    if not model_path or not Path(model_path).exists():
        abort(404, description=f"Model file not found for run '{run_id}'. Was it saved?")

    target_column = run.get("target_column", "")
    task_type = run.get("task_type", "")

    # Resolve input data
    content_type = request.content_type or ""
    if "multipart/form-data" in content_type:
        file = request.files.get("file")
        if file is None:
            abort(400, description="Multipart request must include a 'file' field.")
        try:
            df = pd.read_csv(io.BytesIO(file.read()))
        except Exception as exc:
            abort(400, description=f"Could not parse uploaded CSV: {exc}")
    else:
        body = request.get_json(silent=True) or {}
        rows = body.get("data")
        if not isinstance(rows, list) or not rows:
            abort(400, description="JSON body must contain a non-empty 'data' array of row objects.")
        try:
            df = pd.DataFrame(rows)
        except Exception as exc:
            abort(400, description=f"Could not build DataFrame from data: {exc}")

    # Drop target if accidentally included
    feature_df = df.drop(columns=[target_column], errors="ignore")
    if feature_df.empty or len(feature_df.columns) == 0:
        abort(400, description="No feature columns found in the input data.")

    try:
        pipeline = joblib.load(model_path)
        predictions = pipeline.predict(feature_df)
    except Exception as exc:
        abort(500, description=f"Scoring failed: {exc}")

    preds_list = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)

    result: dict[str, Any] = {
        "run_id": run_id,
        "task_type": task_type,
        "predictions": preds_list,
        "n_rows": len(preds_list),
    }

    if task_type == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(feature_df)
            result["probabilities"] = proba.tolist()
        except Exception:
            pass

    return jsonify(result)


@app.get("/api/projects/<project_id>/autopilot/sessions/<session_id>/notebook")
def download_autopilot_notebook_api(project_id: str, session_id: str):
    store = ProjectStore()
    project = _project_or_404(store, project_id)
    loaded = _load_session_or_404(store, project_id, session_id)
    notebook = build_notebook(project, loaded, store)
    data = serialize_notebook(notebook)
    filename = f"{project_id}_{session_id}.ipynb"
    return Response(
        data,
        mimetype="application/x-ipynb+json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _project_payload(project: ProjectInfo) -> dict[str, Any]:
    payload = asdict(project)
    payload["path"] = str(project.path)
    return payload


def _project_or_404(store: ProjectStore, project_id: str) -> ProjectInfo:
    try:
        return store.get_project(project_id)
    except FileNotFoundError as exc:
        abort(404, description=str(exc))


def _load_session_or_404(store: ProjectStore, project_id: str, session_id: str):
    try:
        return load_session(store, project_id, session_id)
    except FileNotFoundError as exc:
        abort(404, description=str(exc))


def _require_datasets(store: ProjectStore, project_id: str) -> None:
    if not store.list_datasets(project_id):
        abort(400, description="Upload or register at least one dataset before launching AI Autopilot.")


def _resolve_api_key(api_key: str | None) -> str:
    resolved = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not resolved:
        abort(400, description="OpenAI API key is required. Pass api_key or set OPENAI_API_KEY.")
    return resolved


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


def _register_job(job: AutopilotJob) -> None:
    with _jobs_lock:
        _jobs[(job.project_id, job.session_id)] = job


def _remove_job(project_id: str, session_id: str) -> None:
    with _jobs_lock:
        _jobs.pop((project_id, session_id), None)


def _get_job(project_id: str, session_id: str) -> AutopilotJob | None:
    with _jobs_lock:
        return _jobs.get((project_id, session_id))


def _launch_job(job: AutopilotJob, answers: list[str] | None = None) -> None:
    with job.lock:
        if job.worker is not None and job.worker.is_alive():
            abort(409, description="Session is already running.")
        if job.generator is None:
            abort(409, description="Session has no active generator to run.")
        job.status = STATUS_RUNNING
        job.pending_step = None
        job.error = None
        job.should_stop = False
        _mark_autopilot_status(job.autopilot, STATUS_RUNNING)
        job.worker = Thread(target=_drive_job, args=(job, answers), daemon=True)
        job.worker.start()


def _drive_job(job: AutopilotJob, answers: list[str] | None) -> None:
    try:
        generator = job.generator
        if generator is None:
            raise RuntimeError("Autopilot generator is not available.")

        if answers is not None:
            try:
                step = generator.send(answers)
            except StopIteration:
                _finish_job(job, STATUS_COMPLETE)
                return
            if _handle_step(job, step):
                return
            with job.lock:
                if job.should_stop:
                    _finish_job(job, STATUS_IDLE)
                    return

        while True:
            try:
                step = next(generator)
            except StopIteration:
                _finish_job(job, STATUS_COMPLETE)
                return
            if _handle_step(job, step):
                return
            with job.lock:
                if job.should_stop:
                    _finish_job(job, STATUS_IDLE)
                    return
    except Exception as exc:  # defensive runtime boundary
        with job.lock:
            job.status = STATUS_ERROR
            job.error = str(exc)
            job.pending_step = None
        _mark_autopilot_status(job.autopilot, STATUS_ERROR)


def _handle_step(job: AutopilotJob, step: AutopilotStep) -> bool:
    if step.kind != "ask":
        return False
    with job.lock:
        job.status = STATUS_WAITING
        job.pending_step = step_to_jsonable(step)
        job.error = None
    _mark_autopilot_status(job.autopilot, STATUS_WAITING)
    return True


def _finish_job(job: AutopilotJob, status: str) -> None:
    with job.lock:
        job.status = status
        job.pending_step = None
        job.error = None
        job.generator = None
    _mark_autopilot_status(job.autopilot, status)


def _mark_autopilot_status(autopilot: AiAutopilot, status: str) -> None:
    try:
        autopilot.set_status(status)
    except Exception:
        pass


def _job_response(job: AutopilotJob) -> dict[str, Any]:
    with job.lock:
        worker_alive = bool(job.worker and job.worker.is_alive())
        return {
            "project_id": job.project_id,
            "session_id": job.session_id,
            "status": job.status,
            "pending_step": job.pending_step,
            "error": job.error,
            "worker_alive": worker_alive,
            "links": {
                "session": f"/api/projects/{job.project_id}/autopilot/sessions/{job.session_id}",
                "events": f"/api/projects/{job.project_id}/autopilot/sessions/{job.session_id}/events",
                "answers": f"/api/projects/{job.project_id}/autopilot/sessions/{job.session_id}/answers",
                "messages": f"/api/projects/{job.project_id}/autopilot/sessions/{job.session_id}/messages",
                "notebook": f"/api/projects/{job.project_id}/autopilot/sessions/{job.session_id}/notebook",
            },
        }


def _recover_orphaned_session(store: ProjectStore, project_id: str, session_id: str, loaded) -> None:
    """Write STATUS_IDLE to disk for a session whose worker died (e.g. server restart)."""
    writer = SessionWriter(
        store=store,
        project_id=project_id,
        session_id=session_id,
        user_goal=loaded.user_goal,
    )
    writer.set_status(STATUS_IDLE)


def _effective_status(job, loaded) -> str:
    if job is not None:
        return job.status
    if loaded.status in {STATUS_RUNNING, STATUS_WAITING}:
        return STATUS_IDLE
    return loaded.status


def _session_payload(store: ProjectStore, project_id: str, session_id: str) -> dict[str, Any]:
    loaded = _load_session_or_404(store, project_id, session_id)
    job = _get_job(project_id, session_id)
    status = _effective_status(job, loaded)
    pending_step = job.pending_step if job is not None else None
    error = job.error if job is not None else None

    return {
        "project_id": project_id,
        "session_id": loaded.session_id,
        "status": status,
        "user_goal": loaded.user_goal,
        "created_at": loaded.created_at,
        "updated_at": loaded.updated_at,
        "strategy_summary": loaded.strategy_summary,
        "steps": [step_to_jsonable(step) for step in loaded.steps],
        "pending_step": pending_step,
        "error": error,
        "notebook": loaded.notebook,
        "new_datasets": [dataset.to_dict() for dataset in loaded.new_datasets],
        "training_runs": loaded.training_runs,
    }


def _event_stream(store_root: str, project_id: str, session_id: str, from_index: int):
    store = ProjectStore(store_root)
    sent_indexes: set[int] = set()

    while True:
        loaded = load_session(store, project_id, session_id)
        for step in loaded.steps:
            if step.index <= from_index or step.index in sent_indexes:
                continue
            sent_indexes.add(step.index)
            yield _sse("step", step_to_jsonable(step))

        job = _get_job(project_id, session_id)
        status = _effective_status(job, loaded)
        pending_step = job.pending_step if job is not None else None
        error = job.error if job is not None else None

        if status == STATUS_IDLE and loaded.status in {STATUS_RUNNING, STATUS_WAITING} and job is None:
            _recover_orphaned_session(store, project_id, session_id, loaded)
            yield _sse("status", {"status": STATUS_IDLE, "error": None})
            return

        if status == STATUS_WAITING:
            yield _sse("status", {"status": status, "pending_step": pending_step})
            return
        if status in TERMINAL_STATUSES:
            yield _sse("status", {"status": status, "error": error})
            return

        yield _sse("heartbeat", {"status": status})
        time.sleep(0.75)


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"
