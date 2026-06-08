"""aiml-discovery — autonomous AutoML as a Python library.

Quick start::

    from aiml_discovery import Autopilot

    pilot = Autopilot(data="./data.csv", goal="Predict churn")
    for step in pilot.run():
        print(f"[{step.agent}] {step.thought or ''}")

    pilot.save_notebook("./results/notebook.ipynb")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Generator, Iterable

from dotenv import load_dotenv

from backend.logic.agents.base import AutopilotStep
from backend.logic.autopilot import AiAutopilot
from backend.logic.providers import ProviderConfig, provider_from_env
from backend.services.project_store import DatasetInfo, ProjectStore

load_dotenv()

__all__ = ["Autopilot", "AutopilotStep", "ProviderConfig", "run"]
__version__ = "0.1.0"

log = logging.getLogger(__name__)


class Autopilot:
    """High-level wrapper around AiAutopilot for use outside the Flask server.

    Parameters
    ----------
    data:
        Path to a dataset file (CSV, Excel, JSON, SQLite).
    goal:
        Natural-language objective, e.g. "Predict customer churn".
    provider:
        A :class:`ProviderConfig` instance.  Auto-detected from env vars when
        omitted.
    project:
        Label for the run (used to group sessions on disk).  Defaults to the
        data file stem.
    output_dir:
        Directory where the notebook and reports are written after the run.
        Defaults to ``./aiml_output/``.
    session_id:
        Supply a previous session ID to resume an interrupted run.
    """

    def __init__(
        self,
        data: str | Path | None = None,
        goal: str = "",
        provider: ProviderConfig | None = None,
        project: str = "",
        output_dir: str | Path = "./aiml_output",
        session_id: str | None = None,
    ) -> None:
        self._data_path = Path(data).resolve() if data else None
        self._goal = goal
        self._provider = provider or provider_from_env()
        self._output_dir = Path(output_dir).resolve()
        self._project_label = project or (self._data_path.stem if self._data_path else "run")

        self._store = ProjectStore()
        self._project_id = self._ensure_project()

        if self._data_path is not None:
            self._register_dataset()

        self._pilot = AiAutopilot(
            provider_config=self._provider,
            project_id=self._project_id,
            store=self._store,
            user_goal=goal,
            session_id=session_id,
        )

    # ── Public run API ────────────────────────────────────────────────────────

    def run(self) -> Generator[AutopilotStep, None, None]:
        """Run the autopilot and yield each :class:`AutopilotStep` as it completes."""
        yield from self._pilot.run()

    def answer(self, user_message: str) -> Generator[AutopilotStep, None, None]:
        """Resume after the autopilot paused with an ``ask_user`` step."""
        yield from self._pilot.continue_with(user_message)

    def stop(self) -> None:
        """Request a graceful stop after the current agent step."""
        self._pilot.signal_stop()

    # ── Result accessors ──────────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        return self._pilot.session_id

    @property
    def best_model(self) -> str | None:
        """Name of the best-performing model, parsed from training_runs."""
        runs = self.training_runs
        if not runs:
            return None
        metric_key = "test_score" if runs and "test_score" in runs[0] else None
        if metric_key:
            best = max(runs, key=lambda r: r.get(metric_key, float("-inf")), default=None)
            return best.get("model_name") if best else None
        return None

    @property
    def training_runs(self) -> list[dict[str, Any]]:
        return list(self._pilot.training_runs)

    @property
    def strategy_summary(self) -> str:
        return self._pilot.strategy_summary

    @property
    def notebook(self) -> list[str]:
        return list(self._pilot.notebook)

    # ── Output helpers ────────────────────────────────────────────────────────

    def save_notebook(self, path: str | Path | None = None) -> Path:
        """Write the generated Jupyter notebook to *path* and return it."""
        import nbformat

        dest = Path(path) if path else self._output_dir / "notebook.ipynb"
        dest.parent.mkdir(parents=True, exist_ok=True)

        cells: list[nbformat.NotebookNode] = []
        for raw in self.notebook:
            if isinstance(raw, dict):
                cells.append(nbformat.from_dict(raw))
            else:
                try:
                    cells.append(nbformat.from_dict(json.loads(raw)))
                except (json.JSONDecodeError, TypeError):
                    cells.append(nbformat.v4.new_markdown_cell(str(raw)))

        nb = nbformat.v4.new_notebook(cells=cells)
        with dest.open("w", encoding="utf-8") as fh:
            nbformat.write(nb, fh)
        return dest

    def save_results(self, output_dir: str | Path | None = None) -> Path:
        """Save the notebook + training_runs JSON to *output_dir*."""
        dest = Path(output_dir) if output_dir else self._output_dir
        dest.mkdir(parents=True, exist_ok=True)

        self.save_notebook(dest / "notebook.ipynb")

        runs_path = dest / "training_runs.json"
        runs_path.write_text(
            json.dumps(self.training_runs, indent=2, default=str), encoding="utf-8"
        )

        summary_path = dest / "summary.txt"
        summary_path.write_text(self.strategy_summary or "", encoding="utf-8")

        return dest

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_project(self) -> str:
        existing = self._store.list_projects(user_id="local")
        for p in existing:
            if p.name == self._project_label:
                return p.id
        project = self._store.create_project(
            name=self._project_label, description="Created by aiml-discovery CLI"
        )
        return project.id

    def _register_dataset(self) -> None:
        if self._data_path is None or not self._data_path.exists():
            return
        try:
            self._store.register_dataset(
                project_id=self._project_id,
                source_path=self._data_path,
            )
        except Exception:
            pass  # dataset may already be registered


# ── Functional shortcut ───────────────────────────────────────────────────────

def run(
    data: str | Path,
    goal: str,
    *,
    provider: ProviderConfig | None = None,
    output_dir: str | Path = "./aiml_output",
    quiet: bool = False,
) -> Autopilot:
    """Run the autopilot to completion and return the :class:`Autopilot` instance.

    This is a convenience wrapper that blocks until the run finishes::

        pilot = aiml_discovery.run("./data.csv", "Predict churn")
        pilot.save_results("./results")
    """
    pilot = Autopilot(data=data, goal=goal, provider=provider, output_dir=output_dir)
    for step in pilot.run():
        if not quiet and (step.thought or step.tool_call):
            agent = step.agent or "autopilot"
            msg = step.thought or step.tool_call or ""
            print(f"[{agent}] {msg[:120]}")
    return pilot
