"""Per-autopilot-session persistence backed by SQLite (local) or Supabase (hosted).

Replaces the previous file-based approach (JSON/JSONL under PROJECT_HOME).
The public API — SessionWriter, load_session, list_sessions, etc. — is unchanged
so all callers (routes, agents) continue to work without modification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.logic.agents.base import AutopilotStep, to_json_safe
from backend.services.db import get_backend
from backend.services.project_store import DatasetInfo, ProjectStore

log = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# Step <-> JSON helpers  (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────


def step_to_jsonable(step: AutopilotStep) -> dict[str, Any]:
    raw_data = dict(step.data or {})
    figure = raw_data.pop("figure", None)
    if figure is not None:
        try:
            raw_data["figure_json"] = figure.to_json()
        except Exception as exc:
            log.warning("step_to_jsonable | figure.to_json failed: %s", exc)
    return {
        "index": step.index,
        "kind": step.kind,
        "title": step.title,
        "detail": step.detail,
        "data": to_json_safe(raw_data),
        "agent": step.agent,
        "phase": step.phase,
    }


def step_from_jsonable(payload: dict[str, Any]) -> AutopilotStep:
    data = dict(payload.get("data") or {})
    if "figure_json" in data:
        try:
            import plotly.io as pio
            data["figure"] = pio.from_json(data.pop("figure_json"))
        except Exception as exc:
            log.warning("step_from_jsonable | pio.from_json failed: %s", exc)
            data.pop("figure_json", None)
    return AutopilotStep(
        index=int(payload["index"]),
        kind=str(payload["kind"]),
        title=str(payload.get("title", "")),
        detail=str(payload.get("detail", "")),
        data=data,
        agent=str(payload.get("agent", "scientist")),
        phase=str(payload.get("phase", "business_understanding")),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Compatibility shims — session_store used to expose path helpers used by
# old code; keep them pointing at the project cache dir so nothing breaks.
# ──────────────────────────────────────────────────────────────────────────────


def autopilot_root(store: ProjectStore, project_id: str) -> Path:
    return store.project_path(project_id) / "autopilot"


def session_path(store: ProjectStore, project_id: str, session_id: str) -> Path:
    return autopilot_root(store, project_id) / session_id


# ──────────────────────────────────────────────────────────────────────────────
# Session index
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    created_at: str
    updated_at: str
    user_goal: str
    title: str
    status: str
    step_count: int


def list_sessions(store: ProjectStore, project_id: str) -> list[SessionRecord]:
    rows = get_backend().list_sessions(project_id)
    records = [
        SessionRecord(
            session_id=str(r.get("id", "")),
            created_at=str(r.get("created_at", "")),
            updated_at=str(r.get("updated_at", "")),
            user_goal=str(r.get("user_goal", "")),
            title=str(r.get("title") or r.get("user_goal") or r.get("id", "")),
            status=str(r.get("status", "unknown")),
            step_count=int(r.get("step_count") or 0),
        )
        for r in rows
    ]
    records.sort(key=lambda r: r.updated_at, reverse=True)
    return records


def delete_session(store: ProjectStore, project_id: str, session_id: str) -> None:
    get_backend().delete_session(project_id, session_id)


# ──────────────────────────────────────────────────────────────────────────────
# Writer — used during a live run
# ──────────────────────────────────────────────────────────────────────────────


class SessionWriter:
    """Append-only writer backed by the active StorageBackend."""

    def __init__(
        self,
        store: ProjectStore,
        project_id: str,
        session_id: str,
        user_goal: str,
        *,
        created_at: str | None = None,
    ) -> None:
        self.store = store
        self.project_id = project_id
        self.session_id = session_id
        self.user_goal = user_goal
        self._backend = get_backend()

        now = created_at or _utc_now()
        existing = self._backend.get_session(project_id, session_id)
        if existing is None:
            self._backend.upsert_session(
                project_id, session_id,
                user_goal=user_goal,
                title=(user_goal or session_id)[:80],
                status="running",
                strategy_summary="",
                step_count=0,
                created_at=now,
                updated_at=now,
            )
            self._step_count = 0
        else:
            self._backend.upsert_session(project_id, session_id, updated_at=now)
            self._step_count = int(existing.get("step_count") or 0)

    def append_step(self, step: AutopilotStep) -> None:
        try:
            self._backend.append_step(
                self.session_id,
                json.dumps(step_to_jsonable(step), default=str),
            )
            self._step_count += 1
            self._backend.upsert_session(
                self.project_id, self.session_id,
                step_count=self._step_count,
                updated_at=_utc_now(),
            )
        except Exception as exc:
            log.warning("SessionWriter.append_step failed: %s", exc)

    def append_message(self, message: dict[str, Any]) -> None:
        try:
            self._backend.append_message(
                self.session_id,
                json.dumps(to_json_safe(message), default=str),
            )
        except Exception as exc:
            log.warning("SessionWriter.append_message failed: %s", exc)

    def patch_ask_answers(self, step_index: int, answers: list[str]) -> None:
        try:
            patch = {"_patch": True, "index": step_index, "answers": answers}
            self._backend.append_step(self.session_id, json.dumps(patch, default=str))
            self._backend.upsert_session(self.project_id, self.session_id, updated_at=_utc_now())
        except Exception as exc:
            log.warning("SessionWriter.patch_ask_answers failed: %s", exc)

    def save_notebook(self, notebook: list[str]) -> None:
        self._backend.save_notebook(self.session_id, json.dumps(list(notebook), default=str))

    def save_new_datasets(self, datasets: list[DatasetInfo]) -> None:
        self._backend.save_session_datasets(
            self.session_id,
            json.dumps([d.to_dict() for d in datasets], default=str),
        )

    def save_training_runs(self, runs: list[dict[str, Any]]) -> None:
        self._backend.save_session_runs(
            self.session_id,
            json.dumps(to_json_safe(runs), default=str),
        )

    def set_strategy_summary(self, summary: str) -> None:
        self._backend.upsert_session(
            self.project_id, self.session_id,
            strategy_summary=summary,
            updated_at=_utc_now(),
        )

    def set_status(self, status: str) -> None:
        self._backend.upsert_session(
            self.project_id, self.session_id,
            status=status,
            updated_at=_utc_now(),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Reader — used to resume after a page refresh
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class LoadedSession:
    session_id: str
    project_id: str
    user_goal: str
    status: str
    strategy_summary: str
    created_at: str
    updated_at: str
    steps: list[AutopilotStep]
    messages: list[dict[str, Any]]
    notebook: list[str]
    new_datasets: list[DatasetInfo]
    training_runs: list[dict[str, Any]]


def load_session(
    store: ProjectStore, project_id: str, session_id: str
) -> LoadedSession:
    backend = get_backend()
    meta = backend.get_session(project_id, session_id)
    if meta is None:
        raise FileNotFoundError(f"Autopilot session not found: {session_id}")

    raw_steps = [json.loads(s) for s in backend.get_steps(session_id)]
    raw_messages = [json.loads(m) for m in backend.get_messages(session_id)]
    notebook: list[str] = json.loads(backend.get_notebook(session_id))
    new_ds_raw: list[dict] = json.loads(backend.get_session_datasets(session_id))
    runs: list[dict] = json.loads(backend.get_session_runs(session_id))

    # Reconstruct steps, merging patch rows onto their target step.
    by_index: dict[int, dict[str, Any]] = {}
    patches: list[dict[str, Any]] = []
    for row in raw_steps:
        if row.get("_patch"):
            patches.append(row)
        else:
            by_index[int(row["index"])] = row
    for patch in patches:
        idx = int(patch.get("index", -1))
        if idx in by_index:
            data = dict(by_index[idx].get("data") or {})
            data["answers"] = list(patch.get("answers") or [])
            by_index[idx]["data"] = data

    steps = [step_from_jsonable(by_index[i]) for i in sorted(by_index)]

    new_datasets: list[DatasetInfo] = []
    for item in new_ds_raw:
        try:
            new_datasets.append(DatasetInfo(**item))
        except TypeError:
            log.warning("load_session | could not rehydrate DatasetInfo: %s", item)

    return LoadedSession(
        session_id=session_id,
        project_id=project_id,
        user_goal=str(meta.get("user_goal", "")),
        status=str(meta.get("status", "unknown")),
        strategy_summary=str(meta.get("strategy_summary", "")),
        created_at=str(meta.get("created_at", "")),
        updated_at=str(meta.get("updated_at", "")),
        steps=steps,
        messages=raw_messages,
        notebook=list(notebook),
        new_datasets=new_datasets,
        training_runs=list(runs),
    )


def new_session_id(user_goal: str = "") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"session_{stamp}"


__all__ = [
    "LoadedSession",
    "SessionRecord",
    "SessionWriter",
    "autopilot_root",
    "delete_session",
    "list_sessions",
    "load_session",
    "new_session_id",
    "session_path",
    "step_from_jsonable",
    "step_to_jsonable",
]
