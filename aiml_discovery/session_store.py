"""Per-autopilot-session persistence.

Stores everything needed to re-render and continue an AI Autopilot run after
a refresh: the streamed AutopilotStep events, the LLM message history, the
notebook entries, and the snapshots of new datasets and training runs created
during the session.

Layout under each project:

    PROJECT_HOME/{project_id}/autopilot/
        sessions.json                       # index of sessions
        {session_id}/
            session.json                    # metadata + strategy_summary
            steps.jsonl                     # one AutopilotStep per line
            messages.jsonl                  # LLM conversation history
            notebook.json                   # AgentContext.notebook entries
            new_datasets.json               # DatasetInfo dicts produced here
            training_runs.json              # training_run dicts produced here

steps.jsonl and messages.jsonl are append-only so partial runs survive crashes.
Steps with Plotly figures inside ``step.data`` round-trip through
``fig.to_json()``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agents.base import AutopilotStep, to_json_safe
from .storage import DatasetInfo, ProjectStore

log = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# Step <-> JSON helpers
# ──────────────────────────────────────────────────────────────────────────────


def step_to_jsonable(step: AutopilotStep) -> dict[str, Any]:
    """Serialise an AutopilotStep, replacing any live Plotly Figure with its
    JSON spec under ``data["figure_json"]``.
    """
    raw_data = dict(step.data or {})
    figure = raw_data.pop("figure", None)
    if figure is not None:
        try:
            raw_data["figure_json"] = figure.to_json()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("step_to_jsonable | figure.to_json failed: %s", exc)
    return {
        "index": step.index,
        "kind": step.kind,
        "title": step.title,
        "detail": step.detail,
        "data": to_json_safe(raw_data),
        "agent": step.agent,
    }


def step_from_jsonable(payload: dict[str, Any]) -> AutopilotStep:
    """Rebuild an AutopilotStep, re-hydrating any ``figure_json`` back into a
    live Plotly Figure under ``data["figure"]``.
    """
    data = dict(payload.get("data") or {})
    if "figure_json" in data:
        try:
            import plotly.io as pio

            data["figure"] = pio.from_json(data.pop("figure_json"))
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("step_from_jsonable | pio.from_json failed: %s", exc)
            data.pop("figure_json", None)
    return AutopilotStep(
        index=int(payload["index"]),
        kind=str(payload["kind"]),
        title=str(payload.get("title", "")),
        detail=str(payload.get("detail", "")),
        data=data,
        agent=str(payload.get("agent", "scientist")),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ──────────────────────────────────────────────────────────────────────────────


def autopilot_root(store: ProjectStore, project_id: str) -> Path:
    return store.project_path(project_id) / "autopilot"


def session_path(store: ProjectStore, project_id: str, session_id: str) -> Path:
    return autopilot_root(store, project_id) / session_id


def _index_path(store: ProjectStore, project_id: str) -> Path:
    return autopilot_root(store, project_id) / "sessions.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        log.warning("session_store | malformed JSON at %s — using default", path)
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("session_store | bad JSONL line in %s — skipped", path)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Index management
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
    items = _read_json(_index_path(store, project_id), default=[])
    out: list[SessionRecord] = []
    for item in items or []:
        out.append(
            SessionRecord(
                session_id=str(item.get("session_id", "")),
                created_at=str(item.get("created_at", "")),
                updated_at=str(item.get("updated_at", "")),
                user_goal=str(item.get("user_goal", "")),
                title=str(item.get("title", "") or item.get("user_goal", "") or item.get("session_id", "")),
                status=str(item.get("status", "unknown")),
                step_count=int(item.get("step_count", 0) or 0),
            )
        )
    out.sort(key=lambda r: r.updated_at, reverse=True)
    return out


def _upsert_index(
    store: ProjectStore,
    project_id: str,
    session_id: str,
    **updates: Any,
) -> None:
    path = _index_path(store, project_id)
    items: list[dict[str, Any]] = _read_json(path, default=[]) or []
    found = False
    for item in items:
        if item.get("session_id") == session_id:
            item.update(updates)
            found = True
            break
    if not found:
        record = {"session_id": session_id}
        record.update(updates)
        items.append(record)
    _write_json(path, items)


def delete_session(store: ProjectStore, project_id: str, session_id: str) -> None:
    """Remove a session directory and its index entry."""
    import shutil

    path = session_path(store, project_id, session_id)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    items: list[dict[str, Any]] = _read_json(_index_path(store, project_id), default=[]) or []
    items = [item for item in items if item.get("session_id") != session_id]
    _write_json(_index_path(store, project_id), items)


# ──────────────────────────────────────────────────────────────────────────────
# Writer — used during a live run
# ──────────────────────────────────────────────────────────────────────────────


class SessionWriter:
    """Append-only writer attached to an AgentContext and AiAutopilot.

    Persists every step and LLM message as it happens. Safe to call many times;
    each method is a single file append or a small JSON rewrite.
    """

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
        self._dir = session_path(store, project_id, session_id)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._steps_file = self._dir / "steps.jsonl"
        self._messages_file = self._dir / "messages.jsonl"
        self._session_file = self._dir / "session.json"
        self._notebook_file = self._dir / "notebook.json"
        self._new_datasets_file = self._dir / "new_datasets.json"
        self._training_runs_file = self._dir / "training_runs.json"

        # Create/refresh session.json
        now = created_at or _utc_now()
        existing = _read_json(self._session_file, default=None)
        if existing is None:
            payload = {
                "session_id": session_id,
                "project_id": project_id,
                "user_goal": user_goal,
                "status": "running",
                "strategy_summary": "",
                "created_at": now,
                "updated_at": now,
            }
        else:
            payload = dict(existing)
            payload["updated_at"] = now
            payload.setdefault("status", "running")
        _write_json(self._session_file, payload)

        _upsert_index(
            store,
            project_id,
            session_id,
            user_goal=user_goal,
            title=(user_goal or session_id)[:80],
            created_at=payload.get("created_at", now),
            updated_at=payload.get("updated_at", now),
            status=payload.get("status", "running"),
            step_count=payload.get("step_count", 0),
        )

        self._step_count = int(payload.get("step_count", 0))

    # ----- core append API --------------------------------------------

    def append_step(self, step: AutopilotStep) -> None:
        try:
            _append_jsonl(self._steps_file, step_to_jsonable(step))
            self._step_count += 1
            self._touch(step_count=self._step_count)
        except Exception as exc:
            log.warning("SessionWriter.append_step failed: %s", exc)

    def append_message(self, message: dict[str, Any]) -> None:
        try:
            _append_jsonl(self._messages_file, to_json_safe(message))
        except Exception as exc:
            log.warning("SessionWriter.append_message failed: %s", exc)

    def patch_ask_answers(self, step_index: int, answers: list[str]) -> None:
        """Re-write an ``ask`` step so its captured user answers are persisted."""
        try:
            _append_jsonl(
                self._steps_file,
                {"_patch": True, "index": step_index, "answers": answers},
            )
            self._touch()
        except Exception as exc:
            log.warning("SessionWriter.patch_ask_answers failed: %s", exc)

    # ----- snapshots --------------------------------------------------

    def save_notebook(self, notebook: list[str]) -> None:
        _write_json(self._notebook_file, list(notebook))

    def save_new_datasets(self, datasets: list[DatasetInfo]) -> None:
        _write_json(self._new_datasets_file, [d.to_dict() for d in datasets])

    def save_training_runs(self, runs: list[dict[str, Any]]) -> None:
        _write_json(self._training_runs_file, to_json_safe(runs))

    def set_strategy_summary(self, summary: str) -> None:
        meta = _read_json(self._session_file, default={}) or {}
        meta["strategy_summary"] = summary
        meta["updated_at"] = _utc_now()
        _write_json(self._session_file, meta)
        _upsert_index(self.store, self.project_id, self.session_id, updated_at=meta["updated_at"])

    def set_status(self, status: str) -> None:
        meta = _read_json(self._session_file, default={}) or {}
        meta["status"] = status
        meta["updated_at"] = _utc_now()
        _write_json(self._session_file, meta)
        _upsert_index(
            self.store,
            self.project_id,
            self.session_id,
            status=status,
            updated_at=meta["updated_at"],
        )

    # ----- internals --------------------------------------------------

    def _touch(self, **updates: Any) -> None:
        meta = _read_json(self._session_file, default={}) or {}
        meta["updated_at"] = _utc_now()
        for key, value in updates.items():
            meta[key] = value
        _write_json(self._session_file, meta)
        index_updates = {"updated_at": meta["updated_at"]}
        if "step_count" in updates:
            index_updates["step_count"] = updates["step_count"]
        _upsert_index(self.store, self.project_id, self.session_id, **index_updates)


# ──────────────────────────────────────────────────────────────────────────────
# Reader — used to resume after refresh
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
    base = session_path(store, project_id, session_id)
    if not base.exists():
        raise FileNotFoundError(f"Autopilot session not found: {session_id}")

    meta = _read_json(base / "session.json", default={}) or {}
    raw_steps = _read_jsonl(base / "steps.jsonl")
    raw_messages = _read_jsonl(base / "messages.jsonl")
    notebook = _read_json(base / "notebook.json", default=[]) or []
    new_ds_raw = _read_json(base / "new_datasets.json", default=[]) or []
    runs = _read_json(base / "training_runs.json", default=[]) or []

    # Apply ask-answer patches.
    by_index: dict[int, dict[str, Any]] = {}
    patches: list[dict[str, Any]] = []
    for row in raw_steps:
        if row.get("_patch"):
            patches.append(row)
            continue
        by_index[int(row["index"])] = row
    for patch in patches:
        idx = int(patch.get("index", -1))
        if idx in by_index:
            data = dict(by_index[idx].get("data") or {})
            data["answers"] = list(patch.get("answers") or [])
            by_index[idx]["data"] = data

    steps = [step_from_jsonable(by_index[i]) for i in sorted(by_index.keys())]

    new_datasets: list[DatasetInfo] = []
    for item in new_ds_raw:
        try:
            new_datasets.append(DatasetInfo(**item))
        except TypeError:
            # Future schema additions shouldn't break loading old sessions.
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
    "delete_session",
    "list_sessions",
    "load_session",
    "new_session_id",
    "session_path",
    "step_from_jsonable",
    "step_to_jsonable",
]
