"""Autopilot job management: in-memory state for background worker threads."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock, Thread
from typing import Any, Generator

from backend.logic.autopilot import AiAutopilot, AutopilotStep
from backend.services.project_store import ProjectStore
from backend.services.session_store import SessionWriter, step_to_jsonable

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


_jobs: dict[tuple[str, str], AutopilotJob] = {}
_jobs_lock = Lock()


def register_job(job: AutopilotJob) -> None:
    with _jobs_lock:
        _jobs[(job.project_id, job.session_id)] = job


def remove_job(project_id: str, session_id: str) -> None:
    with _jobs_lock:
        _jobs.pop((project_id, session_id), None)


def get_job(project_id: str, session_id: str) -> AutopilotJob | None:
    with _jobs_lock:
        return _jobs.get((project_id, session_id))


def launch_job(job: AutopilotJob, answers: list[str] | None = None) -> None:
    from flask import abort
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


def job_response(job: AutopilotJob) -> dict[str, Any]:
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


def effective_status(job: AutopilotJob | None, loaded) -> str:
    if job is not None:
        return job.status
    if loaded.status in {STATUS_RUNNING, STATUS_WAITING}:
        return STATUS_IDLE
    return loaded.status


def recover_orphaned_session(store: ProjectStore, project_id: str, session_id: str, loaded) -> None:
    """Write STATUS_IDLE to disk for a session whose worker died (e.g. server restart)."""
    writer = SessionWriter(
        store=store,
        project_id=project_id,
        session_id=session_id,
        user_goal=loaded.user_goal,
    )
    writer.set_status(STATUS_IDLE)


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
    except Exception as exc:
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
