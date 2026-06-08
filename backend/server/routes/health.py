from flask import Blueprint, jsonify

from backend.logic.providers import configured_providers, detect_provider_from_env

health_bp = Blueprint("health", __name__)


@health_bp.get("/api/health")
def health():
    detected = detect_provider_from_env()
    configured = configured_providers()
    return jsonify({
        "status": "ok",
        "service": "aiml-discovery-api",
        # Legacy field kept for backward compatibility
        "openai_configured": "openai" in configured or "azure" in configured,
        "detected_provider": detected,
        "configured_providers": configured,
    })
