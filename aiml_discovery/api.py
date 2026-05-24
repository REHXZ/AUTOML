"""HTTP API for AIML Discovery and the AI Autopilot.

Run locally with:

    uvicorn aiml_discovery.api:app --reload

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
from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile, status as http_status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from aiml_discovery.ai_autopilot import AiAutopilot, AutopilotStep
from aiml_discovery.config import APP_NAME
from aiml_discovery.ingestion import load_dataset, list_sqlite_tables
from aiml_discovery.logging_setup import configure_logging
from aiml_discovery.notebook_export import build_notebook, serialize_notebook
from aiml_discovery.session_store import (
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


STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_WAITING = "waiting_for_input"
STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"
TERMINAL_STATUSES = {STATUS_IDLE, STATUS_COMPLETE, STATUS_ERROR}


class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""

    class Config:
        extra = "forbid"


class RegisterDatasetRequest(BaseModel):
    file_path: str = Field(..., min_length=1)
    name: str | None = None
    source_name: str | None = None
    table_name: str | None = None

    class Config:
        extra = "forbid"


class StartAutopilotRequest(BaseModel):
    user_goal: str = ""
    api_key: str | None = Field(default=None, repr=False)

    class Config:
        extra = "forbid"


class AnswerAutopilotRequest(BaseModel):
    answers: list[str] = Field(default_factory=list)

    class Config:
        extra = "forbid"


class FollowUpAutopilotRequest(BaseModel):
    message: str = Field(..., min_length=1)
    api_key: str | None = Field(default=None, repr=False)

    class Config:
        extra = "forbid"


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


app = FastAPI(
    title=f"{APP_NAME} API",
    version="0.1.0",
    description="Local API for AIML Discovery projects, datasets, and AI Autopilot sessions.",
)

_jobs: dict[tuple[str, str], AutopilotJob] = {}
_jobs_lock = Lock()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "aiml-discovery-api",
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
    }


@app.get("/api/projects")
def list_projects_api() -> dict[str, Any]:
    store = ProjectStore()
    return {"projects": [_project_payload(project) for project in store.list_projects()]}


@app.post("/api/projects", status_code=http_status.HTTP_201_CREATED)
def create_project_api(request: CreateProjectRequest) -> dict[str, Any]:
    store = ProjectStore()
    try:
        project = store.create_project(request.name, request.description)
    except ValueError as exc:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"project": _project_payload(project)}


@app.get("/api/projects/{project_id}")
def get_project_api(project_id: str) -> dict[str, Any]:
    store = ProjectStore()
    project = _project_or_404(store, project_id)
    return {"project": _project_payload(project)}


@app.get("/api/projects/{project_id}/datasets")
def list_datasets_api(project_id: str) -> dict[str, Any]:
    store = ProjectStore()
    _project_or_404(store, project_id)
    datasets = store.list_datasets(project_id)
    return {"project_id": project_id, "datasets": [dataset.to_dict() for dataset in datasets]}


@app.post(
    "/api/projects/{project_id}/datasets/upload",
    status_code=http_status.HTTP_201_CREATED,
)
async def upload_dataset_api(
    project_id: str,
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    table_name: str | None = Form(default=None),
) -> dict[str, Any]:
    store = ProjectStore()
    _project_or_404(store, project_id)

    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Dataset filename is required.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Dataset file is empty.",
        )

    saved_path = store.save_dataset_file(project_id, filename, data)
    try:
        datasets = _register_dataset_path(
            store=store,
            project_id=project_id,
            path=saved_path,
            name=name.strip() if name and name.strip() else None,
            source_name=filename,
            table_name=table_name.strip() if table_name and table_name.strip() else None,
        )
    except HTTPException:
        saved_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return {"project_id": project_id, "datasets": [dataset.to_dict() for dataset in datasets]}


@app.post(
    "/api/projects/{project_id}/datasets/register",
    status_code=http_status.HTTP_201_CREATED,
)
def register_dataset_api(
    project_id: str,
    request: RegisterDatasetRequest,
) -> dict[str, Any]:
    store = ProjectStore()
    _project_or_404(store, project_id)

    path = Path(request.file_path).expanduser()
    if not path.exists() or not path.is_file():
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset file not found: {path}",
        )

    datasets = _register_dataset_path(
        store=store,
        project_id=project_id,
        path=path,
        name=request.name,
        source_name=request.source_name,
        table_name=request.table_name,
    )
    return {"project_id": project_id, "datasets": [dataset.to_dict() for dataset in datasets]}


@app.get("/api/projects/{project_id}/autopilot/sessions")
def list_autopilot_sessions_api(project_id: str) -> dict[str, Any]:
    store = ProjectStore()
    _project_or_404(store, project_id)
    return {
        "project_id": project_id,
        "sessions": [asdict(record) for record in list_sessions(store, project_id)],
    }


@app.post(
    "/api/projects/{project_id}/autopilot/sessions",
    status_code=http_status.HTTP_202_ACCEPTED,
)
def start_autopilot_session_api(
    project_id: str,
    request: StartAutopilotRequest,
) -> dict[str, Any]:
    store = ProjectStore()
    _project_or_404(store, project_id)
    _require_datasets(store, project_id)

    api_key = _resolve_api_key(request.api_key)
    autopilot = AiAutopilot(
        api_key=api_key,
        project_id=project_id,
        store=store,
        user_goal=request.user_goal.strip(),
    )
    job = AutopilotJob(
        project_id=project_id,
        session_id=autopilot.session_id,
        autopilot=autopilot,
        generator=autopilot.run(),
    )
    _register_job(job)
    _launch_job(job)

    return _job_response(job)


@app.get("/api/projects/{project_id}/autopilot/sessions/{session_id}")
def get_autopilot_session_api(project_id: str, session_id: str) -> dict[str, Any]:
    store = ProjectStore()
    _project_or_404(store, project_id)
    return _session_payload(store, project_id, session_id)


@app.delete(
    "/api/projects/{project_id}/autopilot/sessions/{session_id}",
    status_code=http_status.HTTP_204_NO_CONTENT,
)
def delete_autopilot_session_api(project_id: str, session_id: str) -> Response:
    store = ProjectStore()
    _project_or_404(store, project_id)

    job = _get_job(project_id, session_id)
    if job is not None and job.worker is not None and job.worker.is_alive():
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail="Cannot delete an autopilot session while it is running.",
        )

    delete_session(store, project_id, session_id)
    _remove_job(project_id, session_id)
    return Response(status_code=http_status.HTTP_204_NO_CONTENT)


@app.get("/api/projects/{project_id}/autopilot/sessions/{session_id}/events")
def stream_autopilot_events_api(
    project_id: str,
    session_id: str,
    from_index: int = Query(0, ge=0),
) -> StreamingResponse:
    store = ProjectStore()
    _project_or_404(store, project_id)
    _load_session_or_404(store, project_id, session_id)

    return StreamingResponse(
        _event_stream(str(store.root), project_id, session_id, from_index),
        media_type="text/event-stream",
    )


@app.post(
    "/api/projects/{project_id}/autopilot/sessions/{session_id}/answers",
    status_code=http_status.HTTP_202_ACCEPTED,
)
def answer_autopilot_questions_api(
    project_id: str,
    session_id: str,
    request: AnswerAutopilotRequest,
) -> dict[str, Any]:
    store = ProjectStore()
    _project_or_404(store, project_id)
    _load_session_or_404(store, project_id, session_id)

    job = _get_job(project_id, session_id)
    if job is None or job.generator is None:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=(
                "This session is not waiting inside the current API process. "
                "Restart the run or continue from a completed session."
            ),
        )
    if job.status != STATUS_WAITING:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=f"Session is not waiting for answers; current status is {job.status}.",
        )

    _launch_job(job, answers=request.answers)
    return _job_response(job)


@app.post(
    "/api/projects/{project_id}/autopilot/sessions/{session_id}/messages",
    status_code=http_status.HTTP_202_ACCEPTED,
)
def continue_autopilot_session_api(
    project_id: str,
    session_id: str,
    request: FollowUpAutopilotRequest,
) -> dict[str, Any]:
    store = ProjectStore()
    _project_or_404(store, project_id)
    loaded = _load_session_or_404(store, project_id, session_id)

    job = _get_job(project_id, session_id)
    if job is not None and job.worker is not None and job.worker.is_alive():
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail="Session is already running.",
        )
    if job is not None and job.status == STATUS_WAITING:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail="Session is waiting for answers. Submit answers before sending a follow-up.",
        )
    if job is None and loaded.status in {STATUS_RUNNING, STATUS_WAITING}:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=(
                "This saved session is mid-run and cannot be resumed without "
                "its active API worker."
            ),
        )

    api_key = _resolve_api_key(request.api_key)
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

    job.generator = job.autopilot.continue_with(request.message.strip())
    _launch_job(job)
    return _job_response(job)


@app.get("/api/projects/{project_id}/autopilot/sessions/{session_id}/notebook")
def download_autopilot_notebook_api(project_id: str, session_id: str) -> Response:
    store = ProjectStore()
    project = _project_or_404(store, project_id)
    loaded = _load_session_or_404(store, project_id, session_id)
    notebook = build_notebook(project, loaded, store)
    data = serialize_notebook(notebook)
    filename = f"{project_id}_{session_id}.ipynb"
    return Response(
        content=data,
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _project_payload(project: ProjectInfo) -> dict[str, Any]:
    payload = asdict(project)
    payload["path"] = str(project.path)
    return payload


def _project_or_404(store: ProjectStore, project_id: str) -> ProjectInfo:
    try:
        return store.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


def _load_session_or_404(store: ProjectStore, project_id: str, session_id: str):
    try:
        return load_session(store, project_id, session_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


def _require_datasets(store: ProjectStore, project_id: str) -> None:
    if not store.list_datasets(project_id):
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Upload or register at least one dataset before launching AI Autopilot.",
        )


def _resolve_api_key(api_key: str | None) -> str:
    resolved = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not resolved:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="OpenAI API key is required. Pass api_key or set OPENAI_API_KEY.",
        )
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
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="No tables were found in the SQLite source.",
            )
    else:
        table_names = [table_name]

    for table in table_names:
        try:
            loaded = load_dataset(path, table_name=table)
        except Exception as exc:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

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


def _launch_job(
    job: AutopilotJob,
    answers: list[str] | None = None,
) -> None:
    with job.lock:
        if job.worker is not None and job.worker.is_alive():
            raise HTTPException(
                status_code=http_status.HTTP_409_CONFLICT,
                detail="Session is already running.",
            )
        if job.generator is None:
            raise HTTPException(
                status_code=http_status.HTTP_409_CONFLICT,
                detail="Session has no active generator to run.",
            )
        job.status = STATUS_RUNNING
        job.pending_step = None
        job.error = None
        _mark_autopilot_status(job.autopilot, STATUS_RUNNING)
        job.worker = Thread(
            target=_drive_job,
            args=(job, answers),
            daemon=True,
        )
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

        while True:
            try:
                step = next(generator)
            except StopIteration:
                _finish_job(job, STATUS_COMPLETE)
                return
            if _handle_step(job, step):
                return
    except Exception as exc:  # pragma: no cover - defensive runtime boundary
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


def _session_payload(
    store: ProjectStore,
    project_id: str,
    session_id: str,
) -> dict[str, Any]:
    loaded = _load_session_or_404(store, project_id, session_id)
    job = _get_job(project_id, session_id)
    status = job.status if job is not None else loaded.status
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


def _event_stream(
    store_root: str,
    project_id: str,
    session_id: str,
    from_index: int,
):
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
        status = job.status if job is not None else loaded.status
        pending_step = job.pending_step if job is not None else None
        error = job.error if job is not None else None

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
