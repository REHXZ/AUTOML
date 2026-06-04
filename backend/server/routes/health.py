import os

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "aiml-discovery-api",
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
    })
