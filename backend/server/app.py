"""Flask application entry point for the AIML Discovery API.

Usage:
    python -m flask --app backend.server.app run --port 8082
    # or
    python backend/server/app.py

Environment variables:
    API_HOST  — bind address (default: 0.0.0.0)
    API_PORT  — port (default: 8082)
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from backend.server.logging_setup import configure_logging
from backend.services.tracing import configure_tracing
from backend.server.routes.health import health_bp
from backend.server.routes.projects import projects_bp
from backend.server.routes.datasets import datasets_bp
from backend.server.routes.providers import providers_bp
from backend.server.routes.sessions import sessions_bp
from backend.server.routes.runs import runs_bp

load_dotenv()
configure_logging()
configure_tracing()

HOST = os.environ.get("API_HOST", "0.0.0.0")
PORT = int(os.environ.get("API_PORT", 8082))

app = Flask(__name__)


@app.errorhandler(HTTPException)
def _http_error(exc: HTTPException):
    return jsonify({"detail": exc.description}), exc.code


app.register_blueprint(health_bp)
app.register_blueprint(projects_bp)
app.register_blueprint(datasets_bp)
app.register_blueprint(providers_bp)
app.register_blueprint(sessions_bp)
app.register_blueprint(runs_bp)


if __name__ == "__main__":
    print(f"Starting AIML Discovery API on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
