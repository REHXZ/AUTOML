import os
from dataclasses import asdict

from flask import Blueprint, Response, abort, jsonify, request, stream_with_context

from backend.logic.autopilot import AiAutopilot
from backend.server.helpers import load_session_or_404, project_or_404
from backend.server.job_manager import (
    STATUS_RUNNING,
    STATUS_WAITING,
    AutopilotJob,
    effective_status,
    get_job,
    job_response,
    launch_job,
    recover_orphaned_session,
    register_job,
    remove_job,
)
from backend.server.streaming import event_stream
from backend.services.project_store import ProjectStore
from backend.services.session_store import (
    delete_session,
    list_sessions,
    step_to_jsonable,
)

sessions_bp = Blueprint("sessions", __name__)


def _require_datasets(store: ProjectStore, project_id: str) -> None:
    if not store.list_datasets(project_id):
        abort(400, description="Upload or register at least one dataset before launching AI Autopilot.")


def _resolve_api_key(api_key: str | None) -> str:
    resolved = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not resolved:
        abort(400, description="OpenAI API key is required. Pass api_key or set OPENAI_API_KEY.")
    return resolved


def _session_payload(store: ProjectStore, project_id: str, session_id: str) -> dict:
    loaded = load_session_or_404(store, project_id, session_id)
    job = get_job(project_id, session_id)
    status = effective_status(job, loaded)
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


@sessions_bp.get("/api/projects/<project_id>/autopilot/sessions")
def list_autopilot_sessions_api(project_id: str):
    store = ProjectStore()
    project_or_404(store, project_id)
    return jsonify({
        "project_id": project_id,
        "sessions": [asdict(r) for r in list_sessions(store, project_id)],
    })


@sessions_bp.post("/api/projects/<project_id>/autopilot/sessions")
def start_autopilot_session_api(project_id: str):
    store = ProjectStore()
    project_or_404(store, project_id)
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
    register_job(job)
    launch_job(job)
    return jsonify(job_response(job)), 202


@sessions_bp.get("/api/projects/<project_id>/autopilot/sessions/<session_id>")
def get_autopilot_session_api(project_id: str, session_id: str):
    store = ProjectStore()
    project_or_404(store, project_id)
    return jsonify(_session_payload(store, project_id, session_id))


@sessions_bp.delete("/api/projects/<project_id>/autopilot/sessions/<session_id>")
def delete_autopilot_session_api(project_id: str, session_id: str):
    store = ProjectStore()
    project_or_404(store, project_id)

    job = get_job(project_id, session_id)
    if job is not None and job.worker is not None and job.worker.is_alive():
        abort(409, description="Cannot delete an autopilot session while it is running.")

    delete_session(store, project_id, session_id)
    remove_job(project_id, session_id)
    return Response("", status=204)


@sessions_bp.get("/api/projects/<project_id>/autopilot/sessions/<session_id>/events")
def stream_autopilot_events_api(project_id: str, session_id: str):
    store = ProjectStore()
    project_or_404(store, project_id)
    load_session_or_404(store, project_id, session_id)

    from_index = max(0, request.args.get("from_index", 0, type=int))
    return Response(
        stream_with_context(
            event_stream(str(store.root), project_id, session_id, from_index)
        ),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@sessions_bp.post("/api/projects/<project_id>/autopilot/sessions/<session_id>/answers")
def answer_autopilot_questions_api(project_id: str, session_id: str):
    store = ProjectStore()
    project_or_404(store, project_id)
    load_session_or_404(store, project_id, session_id)

    job = get_job(project_id, session_id)
    if job is None or job.generator is None:
        abort(409, description=(
            "This session is not waiting inside the current API process. "
            "Restart the run or continue from a completed session."
        ))
    if job.status != STATUS_WAITING:
        abort(409, description=f"Session is not waiting for answers; current status is {job.status}.")

    body = request.get_json(silent=True) or {}
    answers = body.get("answers", [])
    launch_job(job, answers=answers)
    return jsonify(job_response(job)), 202


@sessions_bp.post("/api/projects/<project_id>/autopilot/sessions/<session_id>/messages")
def continue_autopilot_session_api(project_id: str, session_id: str):
    store = ProjectStore()
    project_or_404(store, project_id)
    loaded = load_session_or_404(store, project_id, session_id)

    job = get_job(project_id, session_id)
    if job is not None and job.worker is not None and job.worker.is_alive():
        abort(409, description="Session is already running.")
    if job is not None and job.status == STATUS_WAITING:
        abort(409, description="Session is waiting for answers. Submit answers before sending a follow-up.")
    if job is None and loaded.status == STATUS_WAITING:
        abort(409, description="Session is waiting for answers. Submit answers before sending a follow-up.")
    if job is None and loaded.status == STATUS_RUNNING:
        recover_orphaned_session(store, project_id, session_id, loaded)

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
        register_job(job)

    job.generator = job.autopilot.continue_with(message)
    launch_job(job)
    return jsonify(job_response(job)), 202


@sessions_bp.post("/api/projects/<project_id>/autopilot/sessions/<session_id>/stop")
def stop_autopilot_session_api(project_id: str, session_id: str):
    store = ProjectStore()
    project_or_404(store, project_id)
    job = get_job(project_id, session_id)

    if job is not None and job.worker is not None and job.worker.is_alive():
        job.autopilot.signal_stop()
        with job.lock:
            job.should_stop = True
        return jsonify(job_response(job)), 202

    loaded = load_session_or_404(store, project_id, session_id)
    if loaded.status in {STATUS_RUNNING, STATUS_WAITING}:
        recover_orphaned_session(store, project_id, session_id, loaded)
        return jsonify({
            "project_id": project_id,
            "session_id": session_id,
            "status": "idle",
            "pending_step": None,
            "error": None,
            "worker_alive": False,
        }), 202

    abort(409, description=f"Session is not running (current status: {loaded.status}).")
