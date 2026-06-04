from __future__ import annotations

import ast
from pathlib import Path


def test_api_app_has_valid_python_syntax():
    api_path = Path("backend/server/app.py")

    ast.parse(api_path.read_text(encoding="utf-8"))


def test_api_health_reports_openai_readiness():
    source = Path("backend/server/routes/health.py").read_text(encoding="utf-8")

    assert "openai_configured" in source
    assert "OPENAI_API_KEY" in source


def test_api_exposes_browser_file_upload():
    source = Path("backend/server/routes/datasets.py").read_text(encoding="utf-8")

    assert "datasets/upload" in source
    assert "request.files" in source
