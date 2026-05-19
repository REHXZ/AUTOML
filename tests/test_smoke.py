from __future__ import annotations

import ast
from pathlib import Path


def test_streamlit_app_has_valid_python_syntax():
    app_path = Path("app.py")

    ast.parse(app_path.read_text(encoding="utf-8"))


def test_streamlit_workflow_pages_are_present():
    source = Path("app.py").read_text(encoding="utf-8")

    for page in ["Projects", "Data Sources", "Data Profile", "Training Lab", "Run History", "Model Report"]:
        assert page in source

