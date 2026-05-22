"""Jupyter notebook export for a completed (or interrupted) autopilot session.

Replays an autopilot session into a runnable ``.ipynb``:

  • Markdown cells narrate the reasoning, agent transitions, and findings.
  • Code cells reload datasets via ``pd.read_csv(...)``, materialise plotly
    figures from their persisted JSON specs, and ``joblib.load`` the trained
    models so the recipient can predict immediately.
  • The final strategy report from the Scientist is appended verbatim.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from .agents.base import AutopilotStep
from .session_store import LoadedSession
from .storage import ProjectInfo, ProjectStore

log = logging.getLogger(__name__)


_AGENT_LABELS = {
    "scientist": "AIML Scientist",
    "eda": "EDA Agent",
    "feature_engineering": "Feature Engineering Agent",
    "modeling": "Modeling Agent",
    "review": "Review Agent",
    "fine_tuning": "Fine Tuning Agent",
    "researcher": "Researcher Agent",
}


def _md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


def _code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(src)


def _header_cell(project: ProjectInfo, session: LoadedSession) -> nbformat.NotebookNode:
    parts = [
        f"# AI Autopilot Session — {project.name}",
        "",
        f"- **Session ID:** `{session.session_id}`",
        f"- **Status:** `{session.status}`",
        f"- **Created:** {session.created_at}",
        f"- **Updated:** {session.updated_at}",
    ]
    if session.user_goal:
        parts += ["", "**User goal:**", "", f"> {session.user_goal}"]
    return _md("\n".join(parts))


def _setup_cell() -> nbformat.NotebookNode:
    return _code(
        "# Notebook environment\n"
        "import json\n"
        "import joblib\n"
        "import pandas as pd\n"
        "import plotly.io as pio\n"
    )


def _step_to_cells(step: AutopilotStep) -> list[nbformat.NotebookNode]:
    agent_label = _AGENT_LABELS.get(step.agent, step.agent.title())

    if step.kind == "thought":
        body = step.detail.strip()
        if not body:
            return []
        return [_md(f"### Step {step.index} — {agent_label} reasoning\n\n{body}")]

    if step.kind == "tool_call":
        args_preview = step.detail.strip()
        if len(args_preview) > 600:
            args_preview = args_preview[:600] + "\n…"
        return [_md(
            f"**Step {step.index} — {agent_label} called** `{step.title}`\n\n"
            f"```json\n{args_preview}\n```"
        )]

    if step.kind == "tool_result":
        return [_md(f"**Step {step.index} — Result:** {step.title}\n\n{step.detail}")]

    if step.kind == "chart":
        data = step.data or {}
        figure_json = data.get("figure_json")
        if figure_json is None:
            figure = data.get("figure")
            if figure is not None:
                try:
                    figure_json = figure.to_json()
                except Exception:  # pragma: no cover
                    figure_json = None
        cells: list[nbformat.NotebookNode] = [
            _md(f"### Step {step.index} — Chart: {step.title}\n\n{step.detail}".rstrip())
        ]
        if figure_json:
            literal = json.dumps(figure_json)
            cells.append(_code(
                f"_fig_spec = {literal}\n"
                "fig = pio.from_json(_fig_spec)\n"
                "fig"
            ))
        return cells

    if step.kind == "ask":
        data = step.data or {}
        questions = data.get("questions", []) or []
        answers = data.get("answers", []) or []
        lines = [f"### Step {step.index} — User Q&A"]
        for i, q in enumerate(questions):
            if isinstance(q, dict):
                q_text = q.get("question", "")
                rec = q.get("recommendation", "")
            else:
                q_text = str(q)
                rec = ""
            lines.append(f"\n**Q{i+1}.** {q_text}")
            if rec:
                lines.append(f"\n_Scientist recommended:_ {rec}")
            answer = answers[i] if i < len(answers) else "_(no answer recorded)_"
            lines.append(f"\n**Answer:** {answer}")
        return [_md("\n".join(lines))]

    if step.kind == "new_dataset":
        data = step.data or {}
        ds_id = data.get("dataset_id", "")
        cells = [_md(
            f"### Step {step.index} — New dataset created\n\n"
            f"{step.title} — {step.detail}\n\n"
            f"- dataset_id: `{ds_id}`"
        )]
        return cells

    if step.kind == "training":
        data = step.data or {}
        summary = data.get("summary") or {}
        lines = [f"### Step {step.index} — Training: {step.title}", "", step.detail or ""]
        if summary:
            best = summary.get("best_metrics") or {}
            if best:
                lines.append("")
                lines.append("**Best metrics:**")
                for k, v in best.items():
                    lines.append(f"- {k}: {v}")
        run_id = summary.get("run_id") if isinstance(summary, dict) else None
        cells: list[nbformat.NotebookNode] = [_md("\n".join(lines))]
        if run_id:
            cells.append(_code(
                f"# Load the trained model for run {run_id}\n"
                f"# (Edit the path below if you moved the project store.)\n"
                f"# model = joblib.load(r'PATH_TO_PROJECT/runs/{run_id}/model.joblib')\n"
            ))
        return cells

    if step.kind == "review":
        data = step.data or {}
        lines = [f"### Step {step.index} — Review: {step.title}", "", step.detail or ""]
        if data.get("issues"):
            lines.append("\n**Issues:**")
            lines += [f"- {x}" for x in data["issues"]]
        if data.get("improvements_to_try"):
            lines.append("\n**Improvements to try:**")
            lines += [f"- {x}" for x in data["improvements_to_try"]]
        return [_md("\n".join(lines))]

    if step.kind == "observation":
        return [_md(f"**Step {step.index} — {agent_label} note:** {step.detail}")]

    if step.kind in ("agent_start", "agent_end"):
        bullet = "▶" if step.kind == "agent_start" else "■"
        return [_md(f"## {bullet} Step {step.index} — {step.title}")]

    if step.kind == "summary":
        return [_md(f"## Step {step.index} — Final Strategy Report\n\n{step.detail}")]

    return [_md(f"### Step {step.index} — {step.title}\n\n{step.detail}")]


def _datasets_cell(
    session: LoadedSession, store: ProjectStore
) -> nbformat.NotebookNode | None:
    if not session.new_datasets:
        return None
    lines = ["# Datasets created during this session — load with pandas", ""]
    for ds in session.new_datasets:
        var = ds.name.replace(" ", "_").replace("-", "_") or "df"
        lines.append(f"# {ds.name}: {ds.row_count} rows × {ds.column_count} cols")
        lines.append(f"df_{var} = pd.read_csv(r'{ds.file_path}')")
        lines.append(f"df_{var}.head()")
        lines.append("")
    return _code("\n".join(lines))


def _training_summary_cell(session: LoadedSession) -> nbformat.NotebookNode | None:
    if not session.training_runs:
        return None
    lines = ["## Training runs summary", ""]
    for run in session.training_runs:
        metrics = ", ".join(
            f"{k}={v}" for k, v in (run.get("best_metrics") or {}).items()
        )
        lines.append(
            f"- **{run.get('run_id', '?')}** — dataset `{run.get('dataset')}`,"
            f" target `{run.get('target')}`, task `{run.get('task_type')}`,"
            f" best model `{run.get('best_model')}` ({metrics})"
        )
    return _md("\n".join(lines))


def build_notebook(
    project: ProjectInfo,
    session: LoadedSession,
    store: ProjectStore,
) -> nbformat.NotebookNode:
    nb = new_notebook()
    cells: list[nbformat.NotebookNode] = [
        _header_cell(project, session),
        _setup_cell(),
    ]

    datasets_cell = _datasets_cell(session, store)
    if datasets_cell is not None:
        cells.append(_md("## Generated datasets"))
        cells.append(datasets_cell)

    training_cell = _training_summary_cell(session)
    if training_cell is not None:
        cells.append(training_cell)

    if session.notebook:
        notes = "\n".join(f"- {entry}" for entry in session.notebook)
        cells.append(_md(f"## Shared notebook (agent observations)\n\n{notes}"))

    cells.append(_md("## Step-by-step replay"))
    for step in session.steps:
        cells.extend(_step_to_cells(step))

    if session.strategy_summary:
        cells.append(_md("## Final Strategy Report"))
        cells.append(_md(session.strategy_summary))

    nb["cells"] = cells
    nb["metadata"] = {
        "aiml_autopilot": {
            "project_id": project.id,
            "project_name": project.name,
            "session_id": session.session_id,
            "status": session.status,
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        },
        "language_info": {"name": "python"},
    }
    return nb


def serialize_notebook(notebook: nbformat.NotebookNode) -> bytes:
    return nbformat.writes(notebook).encode("utf-8")


__all__ = ["build_notebook", "serialize_notebook"]
