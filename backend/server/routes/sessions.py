import os
from dataclasses import asdict

from flask import Blueprint, Response, abort, jsonify, request, stream_with_context

from backend.logic.autopilot import AiAutopilot
from backend.logic.providers import ProviderConfig, provider_from_env
from backend.server.auth import get_current_user_id
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


def _resolve_provider_config(body: dict) -> ProviderConfig:
    """Build ProviderConfig from request body, with env-var fallback.

    Accepted body shapes:
      { "provider_config": { "provider": "openai", "api_key": "...", ... } }
      { "api_key": "sk-..." }   ← legacy; treated as openai or azure per env
    """
    raw_cfg = body.get("provider_config")
    if raw_cfg and isinstance(raw_cfg, dict):
        provider = raw_cfg.get("provider", "auto")
        if provider == "auto":
            # Merge env defaults with anything the user explicitly provided.
            # Env wins for api_version: the frontend always sends its hardcoded
            # default ("2024-12-01-preview") which should not override an
            # explicitly configured AZURE_OPENAI_API_VERSION env var.
            env_cfg = provider_from_env()
            provider = env_cfg.provider
            return ProviderConfig(
                provider=provider,
                api_key=raw_cfg.get("api_key") or env_cfg.api_key,
                model=raw_cfg.get("model") or env_cfg.model,
                base_url=raw_cfg.get("base_url") or env_cfg.base_url,
                api_version=env_cfg.api_version or raw_cfg.get("api_version") or "2024-12-01-preview",
            )
        return ProviderConfig(
            provider=provider,
            api_key=raw_cfg.get("api_key", ""),
            model=raw_cfg.get("model", ""),
            base_url=raw_cfg.get("base_url", ""),
            api_version=raw_cfg.get("api_version", "2024-12-01-preview"),
        )

    # Legacy: bare api_key field
    legacy_key = (body.get("api_key") or "").strip()
    env_cfg = provider_from_env()
    if legacy_key:
        env_cfg.api_key = legacy_key

    # Require a key for providers that need one
    if env_cfg.provider not in ("ollama",) and not env_cfg.api_key:
        abort(
            400,
            description=(
                "API key is required. Pass provider_config.api_key in the request body, "
                "or set the appropriate environment variable (OPENAI_API_KEY / ANTHROPIC_API_KEY)."
            ),
        )

    return env_cfg


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
    user_id = get_current_user_id()
    store = ProjectStore()
    project_or_404(store, project_id, user_id=user_id)
    return jsonify({
        "project_id": project_id,
        "sessions": [asdict(r) for r in list_sessions(store, project_id)],
    })


@sessions_bp.post("/api/projects/<project_id>/autopilot/sessions")
def start_autopilot_session_api(project_id: str):
    user_id = get_current_user_id()
    store = ProjectStore()
    project_or_404(store, project_id, user_id=user_id)
    _require_datasets(store, project_id)

    body = request.get_json(silent=True) or {}
    provider_config = _resolve_provider_config(body)
    autopilot = AiAutopilot(
        provider_config=provider_config,
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
    user_id = get_current_user_id()
    store = ProjectStore()
    project_or_404(store, project_id, user_id=user_id)
    return jsonify(_session_payload(store, project_id, session_id))


@sessions_bp.delete("/api/projects/<project_id>/autopilot/sessions/<session_id>")
def delete_autopilot_session_api(project_id: str, session_id: str):
    user_id = get_current_user_id()
    store = ProjectStore()
    project_or_404(store, project_id, user_id=user_id)

    job = get_job(project_id, session_id)
    if job is not None and job.worker is not None and job.worker.is_alive():
        abort(409, description="Cannot delete an autopilot session while it is running.")

    delete_session(store, project_id, session_id)
    remove_job(project_id, session_id)
    return Response("", status=204)


@sessions_bp.get("/api/projects/<project_id>/autopilot/sessions/<session_id>/events")
def stream_autopilot_events_api(project_id: str, session_id: str):
    user_id = get_current_user_id()
    store = ProjectStore()
    project_or_404(store, project_id, user_id=user_id)
    load_session_or_404(store, project_id, session_id)

    from_index = max(0, request.args.get("from_index", 0, type=int))
    return Response(
        stream_with_context(
            event_stream(project_id, session_id, from_index)
        ),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@sessions_bp.post("/api/projects/<project_id>/autopilot/sessions/<session_id>/answers")
def answer_autopilot_questions_api(project_id: str, session_id: str):
    user_id = get_current_user_id()
    store = ProjectStore()
    project_or_404(store, project_id, user_id=user_id)
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
    user_id = get_current_user_id()
    store = ProjectStore()
    project_or_404(store, project_id, user_id=user_id)
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

    provider_config = _resolve_provider_config(body)
    if job is None:
        autopilot = AiAutopilot(
            provider_config=provider_config,
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
    user_id = get_current_user_id()
    store = ProjectStore()
    project_or_404(store, project_id, user_id=user_id)
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
