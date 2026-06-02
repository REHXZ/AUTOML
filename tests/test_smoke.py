from __future__ import annotations

import ast
from pathlib import Path


def test_streamlit_app_has_valid_python_syntax():
    app_path = Path("app.py")

    ast.parse(app_path.read_text(encoding="utf-8"))


def test_api_app_has_valid_python_syntax():
    api_path = Path("aiml_discovery/api.py")

    ast.parse(api_path.read_text(encoding="utf-8"))


def test_api_health_reports_openai_readiness():
    source = Path("aiml_discovery/api.py").read_text(encoding="utf-8")

    assert "openai_configured" in source
    assert "OPENAI_API_KEY" in source


def test_api_exposes_browser_file_upload():
    source = Path("aiml_discovery/api.py").read_text(encoding="utf-8")

    # Flask uses <project_id> placeholders; FastAPI used {project_id}.
    assert "datasets/upload" in source
    assert "request.files" in source or "UploadFile" in source


def test_streamlit_workflow_pages_are_present():
    source = Path("app.py").read_text(encoding="utf-8")

    for page in ["Projects", "Data Sources", "Data Profile", "Training Lab", "Run History", "Model Report"]:
        assert page in source
