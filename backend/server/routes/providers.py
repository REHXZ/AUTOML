"""Provider discovery endpoint — lets the frontend know which providers are pre-configured."""

from flask import Blueprint, jsonify

from backend.logic.providers import (
    PROVIDER_PRESETS,
    configured_providers,
    detect_provider_from_env,
)

providers_bp = Blueprint("providers", __name__)


@providers_bp.get("/api/providers")
def list_providers():
    auto = detect_provider_from_env()
    configured = configured_providers()
    presets = [
        {
            "provider": key,
            "label": val["label"],
            "configured": key in configured,
        }
        for key, val in PROVIDER_PRESETS.items()
    ]
    return jsonify({
        "auto_detected": auto,
        "configured": configured,
        "presets": presets,
    })
