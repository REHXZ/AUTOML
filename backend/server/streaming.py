"""SSE event streaming helpers."""

from __future__ import annotations

import json
import time
from typing import Any

from backend.server.job_manager import (
    TERMINAL_STATUSES,
    STATUS_IDLE,
    STATUS_WAITING,
    effective_status,
    recover_orphaned_session,
    get_job,
)
from backend.services.project_store import ProjectStore
from backend.services.session_store import load_session, step_to_jsonable


def event_stream(project_id: str, session_id: str, from_index: int):
    """Long-polling SSE generator — yields step, status, and heartbeat events."""
    store = ProjectStore()
    sent_indexes: set[int] = set()

    while True:
        loaded = load_session(store, project_id, session_id)
        for step in loaded.steps:
            if step.index <= from_index or step.index in sent_indexes:
                continue
            sent_indexes.add(step.index)
            yield _sse("step", step_to_jsonable(step))

        job = get_job(project_id, session_id)
        status = effective_status(job, loaded)
        pending_step = job.pending_step if job is not None else None
        error = job.error if job is not None else None

        if status == STATUS_IDLE and loaded.status in {STATUS_WAITING, "running"} and job is None:
            recover_orphaned_session(store, project_id, session_id, loaded)
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
