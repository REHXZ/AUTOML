from flask import Blueprint, abort, jsonify, request

from backend.server.auth import get_current_user_id
from backend.server.helpers import project_or_404, project_payload
from backend.services.project_store import ProjectStore

projects_bp = Blueprint("projects", __name__)


@projects_bp.get("/api/projects")
def list_projects_api():
    user_id = get_current_user_id()
    store = ProjectStore()
    return jsonify({"projects": [project_payload(p) for p in store.list_projects(user_id)]})


@projects_bp.post("/api/projects")
def create_project_api():
    user_id = get_current_user_id()
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        abort(400, description="name is required and must be non-empty.")
    store = ProjectStore()
    try:
        project = store.create_project(name, body.get("description", ""), user_id=user_id)
    except ValueError as exc:
        abort(400, description=str(exc))
    return jsonify({"project": project_payload(project)}), 201


@projects_bp.get("/api/projects/<project_id>")
def get_project_api(project_id: str):
    user_id = get_current_user_id()
    store = ProjectStore()
    project = project_or_404(store, project_id, user_id=user_id)
    return jsonify({"project": project_payload(project)})
