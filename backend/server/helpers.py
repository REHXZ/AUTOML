"""Shared route helper utilities."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from flask import abort

from backend.services.project_store import ProjectInfo, ProjectStore
from backend.services.session_store import LoadedSession, load_session


def project_payload(project: ProjectInfo) -> dict[str, Any]:
    payload = asdict(project)
    payload["path"] = str(project.path)
    return payload


def project_or_404(store: ProjectStore, project_id: str, user_id: str | None = None) -> ProjectInfo:
    try:
        return store.get_project(project_id, user_id=user_id)
    except FileNotFoundError as exc:
        abort(404, description=str(exc))


def load_session_or_404(store: ProjectStore, project_id: str, session_id: str) -> LoadedSession:
    try:
        return load_session(store, project_id, session_id)
    except FileNotFoundError as exc:
        abort(404, description=str(exc))
