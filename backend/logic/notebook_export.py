"""Jupyter notebook export for a completed (or interrupted) autopilot session.

Produces a *structured* handover notebook organised by AIML lifecycle phase
(modified CRISP-DM: Business Understanding → Data Understanding → Data
Preparation → Modeling → Evaluation → Iteration). A human can open the
notebook and:

  • read the narrative for each phase,
  • re-run the runnable code cells to reload datasets, re-create derived
    feature tables, and joblib.load the best trained model,
  • inspect the diagnostic charts inline,
  • drop into the full chronological agent transcript in the appendix if
    they want the raw blow-by-blow.

The notebook builder reconstructs runnable code from session artefacts
(``new_datasets``, ``training_runs``, ``store.list_runs``) and from the
arguments captured on the ``tool_call`` / ``new_dataset`` / ``training``
steps. Falls back to ``pd.read_csv`` whenever exact reconstruction is not
possible.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from backend.logic.agents.base import PHASE_BY_ID, PHASE_IDS, PHASES, AutopilotStep
from backend.services.session_store import LoadedSession
from backend.services.project_store import DatasetInfo, ProjectInfo, ProjectStore

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


# ──────────────────────────────────────────────────────────────────────────────
# Header / setup
# ──────────────────────────────────────────────────────────────────────────────


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


def _lifecycle_overview_cell() -> nbformat.NotebookNode:
    lines = [
        "## How this notebook is organised",
        "",
        "This is a handover notebook structured around the modified CRISP-DM",
        "AIML lifecycle. Each section below corresponds to a lifecycle phase",
        "and contains a narrative of what the agents did, runnable code cells",
        "that re-create the work, and any diagnostic charts produced.",
        "",
    ]
    for i, phase in enumerate(PHASES, start=1):
        lines.append(f"{i}. **{phase['title']}** — {phase['description']}")
    lines += [
        "",
        "The full chronological agent transcript lives in the appendix at the",
        "bottom of the notebook if you need to audit individual agent calls.",
    ]
    return _md("\n".join(lines))


def _setup_cell() -> nbformat.NotebookNode:
    return _code(
        "# Environment setup — re-run before any phase below.\n"
        "import json\n"
        "import joblib\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import plotly.express as px\n"
        "import plotly.io as pio\n"
        "\n"
        "pd.set_option('display.max_columns', 80)\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — pulling info out of step.data
# ──────────────────────────────────────────────────────────────────────────────


def _safe_data(step: AutopilotStep) -> dict[str, Any]:
    return dict(step.data or {})


def _parse_tool_args(step: AutopilotStep) -> dict[str, Any]:
    """Recover the JSON args from a tool_call step's detail string."""
    try:
        return json.loads(step.detail)
    except (json.JSONDecodeError, ValueError):
        return {}


def _figure_from_step(step: AutopilotStep) -> str | None:
    """Return the figure JSON for a step that contains one, or None."""
    data = _safe_data(step)
    if "figure_json" in data:
        return data["figure_json"]
    fig = data.get("figure")
    if fig is None:
        return None
    try:
        return fig.to_json()
    except Exception:  # pragma: no cover
        return None


def _python_repr(value: Any) -> str:
    """Render a value as a Python literal safe to paste into a code cell."""
    try:
        return repr(value)
    except Exception:
        return "None"


def _chart_code(step: AutopilotStep) -> str | None:
    """Reconstruct a runnable plotly-express snippet for an EDA chart step.

    Returns None when chart_type/params weren't recorded (older sessions);
    the caller should fall back to ``pio.from_json`` in that case.
    """
    data = _safe_data(step)
    chart_type = data.get("chart_type")
    if not chart_type:
        return None
    params: dict[str, Any] = data.get("chart_params") or {}
    dataset_name = data.get("dataset_name") or "df"
    df = f"df_{_safe_varname(dataset_name)}"

    def _q(value: Any) -> str:
        return repr(value)

    if chart_type == "histogram":
        col = params.get("column")
        bins = int(params.get("bins", 30))
        return (
            f"# Histogram of {col}\n"
            f"fig = px.histogram(\n"
            f"    {df}, x={_q(col)}, nbins={bins},\n"
            f"    title={_q(f'Distribution of {col}')},\n"
            f"    template='plotly_white',\n"
            f")\n"
            f"fig"
        )

    if chart_type == "bar":
        col = params.get("column")
        top_n = int(params.get("top_n", 20))
        return (
            f"# Top-{top_n} value counts of {col}\n"
            f"_vc = {df}[{_q(col)}].value_counts().head({top_n}).reset_index()\n"
            f"_vc.columns = ['value', 'count']\n"
            f"fig = px.bar(\n"
            f"    _vc, x='value', y='count',\n"
            f"    title={_q(f'Value Counts: {col} (top {top_n})')},\n"
            f"    template='plotly_white',\n"
            f"    labels={{'value': {_q(col)}}},\n"
            f")\n"
            f"fig"
        )

    if chart_type == "scatter":
        x_col = params.get("x_column")
        y_col = params.get("y_column")
        color_col = params.get("color_column")
        color_arg = f"\n    color={_q(color_col)}," if color_col else ""
        return (
            f"# Scatter of {x_col} vs {y_col}{f' coloured by {color_col}' if color_col else ''}\n"
            f"fig = px.scatter(\n"
            f"    {df}.head(2000), x={_q(x_col)}, y={_q(y_col)},{color_arg}\n"
            f"    title={_q(f'Scatter: {x_col} vs {y_col}')},\n"
            f"    template='plotly_white', opacity=0.6,\n"
            f")\n"
            f"fig"
        )

    if chart_type == "correlation_heatmap":
        return (
            f"# Correlation heatmap across numeric columns of {dataset_name}\n"
            f"_num = {df}.select_dtypes(include='number')\n"
            f"_corr = _num.corr()\n"
            f"fig = px.imshow(\n"
            f"    _corr, title={_q(f'Correlation — {dataset_name}')},\n"
            f"    color_continuous_scale='RdBu_r', zmin=-1, zmax=1,\n"
            f"    text_auto='.2f', template='plotly_white',\n"
            f")\n"
            f"fig"
        )

    if chart_type == "box":
        col = params.get("column")
        gb = params.get("group_by")
        x_arg = f", x={_q(gb)}" if gb else ""
        comment = f"Box plot of {col}" + (f" grouped by {gb}" if gb else "")
        return (
            f"# {comment}\n"
            f"fig = px.box(\n"
            f"    {df}, y={_q(col)}{x_arg},\n"
            f"    title={_q(f'Box Plot: {col}')},\n"
            f"    template='plotly_white',\n"
            f")\n"
            f"fig"
        )

    if chart_type == "violin":
        col = params.get("column")
        gb = params.get("group_by")
        x_arg = f", x={_q(gb)}" if gb else ""
        comment = f"Violin plot of {col}" + (f" grouped by {gb}" if gb else "")
        return (
            f"# {comment}\n"
            f"fig = px.violin(\n"
            f"    {df}, y={_q(col)}{x_arg},\n"
            f"    box=True, points='outliers',\n"
            f"    title={_q(f'Violin: {col}')},\n"
            f"    template='plotly_white',\n"
            f")\n"
            f"fig"
        )

    if chart_type == "pairplot":
        cols = list(params.get("columns") or [])
        color_col = params.get("color_column")
        color_arg = f"\n    color={_q(color_col)}," if color_col else ""
        return (
            f"# Pairplot across {cols}\n"
            f"_cols = {cols!r}\n"
            f"fig = px.scatter_matrix(\n"
            f"    {df}.head(2000)[_cols],\n"
            f"    dimensions=_cols,{color_arg}\n"
            f"    title='Pairplot: ' + ', '.join(_cols),\n"
            f"    template='plotly_white',\n"
            f")\n"
            f"fig.update_traces(diagonal=dict(visible=False), showupperhalf=False)\n"
            f"fig"
        )

    if chart_type == "missing_heatmap":
        return (
            f"# Missing-value pattern in {dataset_name} (first 100 rows)\n"
            f"_mask = {df}.isnull()\n"
            f"_missing_cols = _mask.columns[_mask.any()].tolist()\n"
            f"_sample = _mask[_missing_cols].head(100).astype(int)\n"
            f"fig = px.imshow(\n"
            f"    _sample.T,\n"
            f"    title={_q(f'Missing Values — {dataset_name} (first 100 rows)')},\n"
            f"    color_continuous_scale=[[0, '#f0f0f0'], [1, '#e53e3e']],\n"
            f"    labels={{'x': 'Row', 'y': 'Column', 'color': 'Missing'}},\n"
            f"    template='plotly_white',\n"
            f")\n"
            f"fig"
        )

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Narrative builders
# ──────────────────────────────────────────────────────────────────────────────


def _phase_narrative(
    phase: dict[str, str], steps: list[AutopilotStep]
) -> nbformat.NotebookNode:
    """Build a markdown summary cell for one lifecycle phase."""
    lines = [f"## {phase['title']}", "", f"_{phase['description']}_", ""]

    # Phase transition rationales (often more useful than raw thoughts).
    transitions = [s for s in steps if s.kind == "phase_transition"]
    for tr in transitions:
        data = _safe_data(tr)
        rationale = data.get("rationale") or tr.detail
        if rationale:
            lines.append(f"> **Why this phase:** {rationale}")
            lines.append("")

    # Which agents worked in this phase.
    agents_used: list[str] = []
    for s in steps:
        if s.kind == "agent_start":
            label = _AGENT_LABELS.get(s.agent, s.agent.title())
            if label not in agents_used:
                agents_used.append(label)
    if agents_used:
        lines.append("**Agents involved:** " + ", ".join(agents_used))
        lines.append("")

    # Observations — the running narrative for this phase.
    observations = [s for s in steps if s.kind == "observation"]
    if observations:
        lines.append("**Notebook observations recorded in this phase:**")
        lines.append("")
        for obs in observations:
            label = _AGENT_LABELS.get(obs.agent, obs.agent.title())
            note = obs.detail.strip() or obs.title
            if note:
                lines.append(f"- _{label}_: {note}")
        lines.append("")

    # Review issues / improvements when this is the evaluation phase.
    reviews = [s for s in steps if s.kind == "review"]
    if reviews:
        lines.append("**Review findings:**")
        lines.append("")
        for rv in reviews:
            data = _safe_data(rv)
            if data.get("issues"):
                lines.append("- Issues:")
                lines += [f"    - {x}" for x in data["issues"]]
            if data.get("improvements_to_try"):
                lines.append("- Improvements to try:")
                lines += [f"    - {x}" for x in data["improvements_to_try"]]
        lines.append("")

    # User Q&A in this phase.
    asks = [s for s in steps if s.kind == "ask"]
    if asks:
        lines.append("**User Q&A in this phase:**")
        lines.append("")
        for ask in asks:
            data = _safe_data(ask)
            questions = data.get("questions") or []
            answers = data.get("answers") or []
            for i, q in enumerate(questions):
                if isinstance(q, dict):
                    q_text = q.get("question", "")
                    rec = q.get("recommendation", "")
                else:
                    q_text = str(q)
                    rec = ""
                lines.append(f"- **Q:** {q_text}")
                if rec:
                    lines.append(f"  - _Recommended:_ {rec}")
                ans = answers[i] if i < len(answers) else ""
                if ans:
                    lines.append(f"  - **Answer:** {ans}")
        lines.append("")

    if len(lines) <= 4:
        lines.append("_(no agent activity recorded in this phase)_")

    return _md("\n".join(lines).rstrip())


def _phase_charts(steps: list[AutopilotStep]) -> list[nbformat.NotebookNode]:
    """Replay every chart yielded inside this phase as runnable cells.

    Prefers a readable ``px.<kind>(...)`` snippet built from the captured
    chart_type/params. Falls back to ``pio.from_json`` only when those
    params weren't recorded (older sessions).
    """
    out: list[nbformat.NotebookNode] = []
    for step in steps:
        if step.kind != "chart":
            continue
        title = step.title or "Chart"
        out.append(_md(f"### Chart: {title}\n\n{step.detail}".rstrip()))

        code = _chart_code(step)
        if code is not None:
            out.append(_code(code))
            continue

        figure_json = _figure_from_step(step)
        if figure_json:
            literal = json.dumps(figure_json)
            out.append(_code(
                "# Chart params weren't captured for this step — "
                "rendering from the saved Plotly JSON.\n"
                f"_fig_spec = {literal}\n"
                "fig = pio.from_json(_fig_spec)\n"
                "fig"
            ))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Phase-specific runnable code
# ──────────────────────────────────────────────────────────────────────────────


def _safe_varname(name: str) -> str:
    cleaned = "".join(c if c.isalnum() else "_" for c in name).strip("_") or "df"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _business_understanding_code(
    session: LoadedSession,
) -> list[nbformat.NotebookNode]:
    if not session.user_goal:
        return []
    return [_md(
        "**Problem statement (from the user):**\n\n"
        f"> {session.user_goal}"
    )]


def _data_understanding_code(
    session: LoadedSession,
    store: ProjectStore,
    project: ProjectInfo,
) -> list[nbformat.NotebookNode]:
    """Re-load each ORIGINAL dataset and show shape + head + describe."""
    originals = store.list_datasets(project.id) or []
    # Filter out anything that was created during this session (those belong
    # to data_preparation). Match by id.
    new_ids = {d.id for d in session.new_datasets}
    originals = [d for d in originals if d.id not in new_ids]
    if not originals:
        return []

    cells: list[nbformat.NotebookNode] = [_md(
        "### Reload the source datasets\n\n"
        "These are the datasets that existed when the session started."
    )]
    for ds in originals:
        var = _safe_varname(ds.name)
        cells.append(_code(
            f"# Source dataset: {ds.name} — {ds.row_count:,} rows × {ds.column_count} cols\n"
            f"df_{var} = pd.read_csv(r{ds.file_path!r})\n"
            f"print(df_{var}.shape)\n"
            f"df_{var}.head()"
        ))
        cells.append(_code(f"df_{var}.describe(include='all').T"))
    return cells


def _data_preparation_code(
    session: LoadedSession,
    steps: list[AutopilotStep],
) -> list[nbformat.NotebookNode]:
    """Show every derived dataset that was created in this phase, with the
    operation that created it (as captured on the tool_call step) and a
    pd.read_csv to load the materialised CSV.
    """
    if not session.new_datasets:
        return []

    # Build an index: dataset_id → operation/rationale gleaned from
    # `new_dataset` steps so we can attach context to each dataset cell.
    new_ds_steps = {
        _safe_data(s).get("dataset_id"): s
        for s in steps
        if s.kind == "new_dataset"
    }
    # Also collect FE tool_call args keyed by output dataset name when
    # available — the args contain the params we'd need to truly replay the
    # transformation in pandas.
    fe_tool_calls: list[dict[str, Any]] = []
    for s in steps:
        if s.kind == "tool_call" and "create_derived_dataset" in s.title:
            args = _parse_tool_args(s)
            if args:
                fe_tool_calls.append(args)

    cells: list[nbformat.NotebookNode] = [_md(
        "### Derived datasets created in this phase\n\n"
        "Each block below corresponds to one Feature Engineering transformation. "
        "The materialised CSV is loaded with `pd.read_csv` for convenience; the "
        "original transformation parameters are shown above each load so you can "
        "re-create the operation in your own pipeline if needed."
    )]

    for ds in session.new_datasets:
        step = new_ds_steps.get(ds.id)
        op = ""
        rationale = ""
        if step is not None:
            data = _safe_data(step)
            op = data.get("operation") or ""
            rationale = data.get("rationale") or ""

        # Find the matching tool_call args by new_name to surface params.
        params_repr = "(params not captured)"
        matched_args: dict[str, Any] = {}
        for args in fe_tool_calls:
            if args.get("new_name") == ds.name:
                matched_args = args
                break
        if matched_args:
            params_repr = json.dumps(matched_args, indent=2)

        notes = []
        if op:
            notes.append(f"- **Operation:** `{op}`")
        if rationale:
            notes.append(f"- **Rationale:** {rationale}")
        notes.append(f"- **Resulting shape:** {ds.row_count:,} rows × {ds.column_count} cols")
        cells.append(_md(
            f"#### `{ds.name}`\n\n"
            + "\n".join(notes)
            + "\n\n<details><summary>Show original FE call (params)</summary>\n\n"
            + f"```json\n{params_repr}\n```\n\n</details>"
        ))
        var = _safe_varname(ds.name)
        cells.append(_code(
            f"df_{var} = pd.read_csv(r{ds.file_path!r})\n"
            f"print(df_{var}.shape)\n"
            f"df_{var}.head()"
        ))
    return cells


def _modeling_code(
    session: LoadedSession,
    steps: list[AutopilotStep],
    store: ProjectStore,
    project: ProjectInfo,
) -> list[nbformat.NotebookNode]:
    """For each training run produced in this phase, emit code that reloads
    the dataset, loads the saved model, and runs prediction on the test split.
    """
    if not session.training_runs:
        return []

    # Index full run metadata (including model_path) from the store.
    all_runs = store.list_runs(project.id) or []
    runs_by_id = {r.get("run_id"): r for r in all_runs}
    datasets_by_id = {d.id: d for d in store.list_datasets(project.id) or []}
    # Also include datasets created in this session (they may not be in the
    # global list if the session is unfinished).
    for d in session.new_datasets:
        datasets_by_id.setdefault(d.id, d)

    cells: list[nbformat.NotebookNode] = [_md(
        "### Training runs in this phase\n\n"
        "Each run below can be reloaded with `joblib.load`. The code re-creates "
        "the same train/test split the Modeling Agent used so you can sanity-check "
        "the metrics yourself or fit additional models on identical data."
    )]

    for run in session.training_runs:
        run_id = run.get("run_id", "")
        full = runs_by_id.get(run_id, {})
        target = run.get("target") or full.get("target_column", "")
        task = run.get("task_type") or full.get("task_type", "")
        best_model = run.get("best_model") or full.get("best_model_name", "")
        metrics = run.get("best_metrics") or full.get("best_metrics") or {}
        dataset_id = run.get("dataset_id") or (full.get("dataset") or {}).get("id", "")
        time_column = run.get("time_column") or (full.get("settings") or {}).get("time_column")
        test_size = (full.get("settings") or {}).get("test_size", 0.2)
        random_state = (full.get("settings") or {}).get("random_state", 42)
        model_path = full.get("model_path", "")

        ds = datasets_by_id.get(dataset_id)
        dataset_label = ds.name if ds else run.get("dataset", "(unknown)")
        var = _safe_varname(dataset_label)

        metrics_md = ", ".join(f"`{k}={v}`" for k, v in metrics.items()) or "_(none)_"
        header = [
            f"#### Run `{run_id}` — {dataset_label}",
            "",
            f"- **Target:** `{target}`",
            f"- **Task:** `{task}`",
            f"- **Best model:** `{best_model}`",
            f"- **Metrics:** {metrics_md}",
        ]
        if time_column:
            header.append(
                f"- **Split:** chronological by `{time_column}` "
                f"(last {test_size:.0%} of rows = test set)"
            )
        else:
            header.append(
                f"- **Split:** random (`test_size={test_size}`, "
                f"`random_state={random_state}`)"
            )
        cells.append(_md("\n".join(header)))

        if not ds:
            cells.append(_md(
                "_Dataset could not be located in the project store — "
                "skipping reload code._"
            ))
            continue

        # Build the reproduction code cell.
        lines = [
            f"# Reload dataset and recreate the split used for run {run_id}",
            f"df_{var} = pd.read_csv(r{ds.file_path!r})",
        ]
        if time_column:
            lines += [
                f"df_{var} = df_{var}.sort_values({time_column!r}).reset_index(drop=True)",
                f"_n_test = int(len(df_{var}) * {test_size})",
                f"_train = df_{var}.iloc[:-_n_test]",
                f"_test = df_{var}.iloc[-_n_test:]",
            ]
        else:
            lines += [
                "from sklearn.model_selection import train_test_split",
                f"_train, _test = train_test_split(df_{var}, "
                f"test_size={test_size}, random_state={random_state})",
            ]
        if target:
            lines += [
                f"X_test = _test.drop(columns=[{target!r}])",
                f"y_test = _test[{target!r}]",
            ]
        cells.append(_code("\n".join(lines)))

        if model_path:
            cells.append(_code(
                f"# Load the trained model for run {run_id}\n"
                f"model = joblib.load(r{model_path!r})\n"
                "model"
            ))
            if target:
                cells.append(_code(
                    f"# Score on the held-out test set\n"
                    f"preds = model.predict(X_test)\n"
                    f"pd.DataFrame({{'y_true': y_test.values[:20], "
                    f"'y_pred': preds[:20]}})"
                ))
        else:
            cells.append(_md(
                f"_Model path for run `{run_id}` was not recorded — "
                "cannot auto-load._"
            ))
    return cells


def _evaluation_code(
    session: LoadedSession,
) -> list[nbformat.NotebookNode]:
    """A DataFrame comparing every training run side-by-side."""
    if not session.training_runs:
        return []
    cells: list[nbformat.NotebookNode] = [_md(
        "### Run leaderboard\n\n"
        "All training runs from this session, side-by-side."
    )]
    rows_repr = json.dumps(
        [
            {
                "run_id": r.get("run_id", ""),
                "dataset": r.get("dataset", ""),
                "target": r.get("target", ""),
                "task_type": r.get("task_type", ""),
                "best_model": r.get("best_model", ""),
                "split_mode": r.get("split_mode", ""),
                **{k: v for k, v in (r.get("best_metrics") or {}).items()},
            }
            for r in session.training_runs
        ],
        indent=2,
    )
    cells.append(_code(
        f"_runs = {rows_repr}\n"
        "leaderboard = pd.DataFrame(_runs)\n"
        "leaderboard"
    ))
    return cells


# ──────────────────────────────────────────────────────────────────────────────
# Per-phase section assembly
# ──────────────────────────────────────────────────────────────────────────────


def _phase_section(
    phase: dict[str, str],
    steps: list[AutopilotStep],
    session: LoadedSession,
    store: ProjectStore,
    project: ProjectInfo,
) -> list[nbformat.NotebookNode]:
    cells: list[nbformat.NotebookNode] = [_md("---")]
    cells.append(_phase_narrative(phase, steps))

    pid = phase["id"]
    if pid == "business_understanding":
        cells.extend(_business_understanding_code(session))
    elif pid == "data_understanding":
        cells.extend(_data_understanding_code(session, store, project))
    elif pid == "data_preparation":
        cells.extend(_data_preparation_code(session, steps))
    elif pid == "modeling":
        cells.extend(_modeling_code(session, steps, store, project))
    elif pid == "evaluation":
        cells.extend(_evaluation_code(session))
    # `iteration` is narrative-only — no reproducible code to emit.

    cells.extend(_phase_charts(steps))
    return cells


# ──────────────────────────────────────────────────────────────────────────────
# Appendix — full chronological transcript (the old behaviour, demoted)
# ──────────────────────────────────────────────────────────────────────────────


def _transcript_cells_for_step(step: AutopilotStep) -> list[nbformat.NotebookNode]:
    agent_label = _AGENT_LABELS.get(step.agent, step.agent.title())
    phase_meta = PHASE_BY_ID.get(step.phase, {})
    phase_label = phase_meta.get("title", step.phase)
    prefix = f"Step {step.index} · _{phase_label}_ · **{agent_label}**"

    if step.kind == "phase_transition":
        data = _safe_data(step)
        return [_md(
            f"{prefix} — **Phase transition** "
            f"(`{data.get('from_phase', '?')}` → `{data.get('to_phase', '?')}`)\n\n"
            f"{step.detail}"
        )]

    if step.kind == "thought":
        body = step.detail.strip()
        if not body:
            return []
        return [_md(f"{prefix} — _reasoning_\n\n{body}")]

    if step.kind in ("tool_call", "tool_result"):
        # Suppressed in the human-readable transcript — the phase narratives
        # above already capture what each agent did, and the raw tool args
        # / results are mostly noise for a reader.
        return []

    if step.kind == "chart":
        cells: list[nbformat.NotebookNode] = [
            _md(f"{prefix} — chart: {step.title}\n\n{step.detail}".rstrip())
        ]
        code = _chart_code(step)
        if code is not None:
            cells.append(_code(code))
            return cells
        figure_json = _figure_from_step(step)
        if figure_json:
            literal = json.dumps(figure_json)
            cells.append(_code(
                "# Chart params weren't captured for this step — "
                "rendering from the saved Plotly JSON.\n"
                f"_fig_spec = {literal}\nfig = pio.from_json(_fig_spec)\nfig"
            ))
        return cells

    if step.kind == "ask":
        data = _safe_data(step)
        questions = data.get("questions") or []
        answers = data.get("answers") or []
        lines = [f"{prefix} — User Q&A"]
        for i, q in enumerate(questions):
            if isinstance(q, dict):
                q_text = q.get("question", "")
                rec = q.get("recommendation", "")
            else:
                q_text = str(q)
                rec = ""
            lines.append(f"\n**Q{i+1}.** {q_text}")
            if rec:
                lines.append(f"\n_Recommended:_ {rec}")
            answer = answers[i] if i < len(answers) else "_(no answer recorded)_"
            lines.append(f"\n**Answer:** {answer}")
        return [_md("\n".join(lines))]

    if step.kind == "new_dataset":
        data = _safe_data(step)
        return [_md(
            f"{prefix} — new dataset: {step.title}\n\n"
            f"{step.detail}\n\n- dataset_id: `{data.get('dataset_id', '')}`"
        )]

    if step.kind == "training":
        data = _safe_data(step)
        summary = data.get("summary") or data or {}
        lines = [f"{prefix} — training: {step.title}", "", step.detail or ""]
        best = summary.get("best_metrics") if isinstance(summary, dict) else None
        if best:
            lines.append("")
            lines.append("**Best metrics:**")
            for k, v in best.items():
                lines.append(f"- {k}: {v}")
        return [_md("\n".join(lines))]

    if step.kind == "review":
        data = _safe_data(step)
        lines = [f"{prefix} — review: {step.title}", "", step.detail or ""]
        if data.get("issues"):
            lines.append("\n**Issues:**")
            lines += [f"- {x}" for x in data["issues"]]
        if data.get("improvements_to_try"):
            lines.append("\n**Improvements to try:**")
            lines += [f"- {x}" for x in data["improvements_to_try"]]
        return [_md("\n".join(lines))]

    if step.kind == "observation":
        return [_md(f"{prefix} — note: {step.detail}")]

    if step.kind in ("agent_start", "agent_end"):
        # Agent lifecycle markers — not useful in the transcript; the phase
        # narratives list which agents participated.
        return []

    if step.kind == "summary":
        return [_md(f"{prefix} — final strategy\n\n{step.detail}")]

    return [_md(f"{prefix} — {step.title}\n\n{step.detail}")]


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────


def build_notebook(
    project: ProjectInfo,
    session: LoadedSession,
    store: ProjectStore,
) -> nbformat.NotebookNode:
    nb = new_notebook()

    cells: list[nbformat.NotebookNode] = [
        _header_cell(project, session),
        _lifecycle_overview_cell(),
        _setup_cell(),
    ]

    # Bucket every step by phase. Anything with an unknown phase falls into
    # the first phase so it still shows up somewhere.
    by_phase: dict[str, list[AutopilotStep]] = {pid: [] for pid in PHASE_IDS}
    for step in session.steps:
        bucket = step.phase if step.phase in by_phase else PHASE_IDS[0]
        by_phase[bucket].append(step)

    # Emit one section per phase, in lifecycle order. Skip phases that have
    # zero activity AND no reproducible artefacts.
    for phase in PHASES:
        pid = phase["id"]
        steps = by_phase.get(pid, [])
        # Skip empty narrative phases unless artefacts exist for them.
        if not steps and pid not in {"data_understanding", "evaluation"}:
            continue
        cells.extend(_phase_section(phase, steps, session, store, project))

    if session.strategy_summary:
        cells.append(_md("---"))
        cells.append(_md(f"## Final Strategy Report\n\n{session.strategy_summary}"))

    # Appendix — full chronological transcript.
    cells.append(_md("---"))
    cells.append(_md(
        "## Appendix — Full agent transcript\n\n"
        "Chronological replay of every agent step in this session. Useful for "
        "auditing the run or debugging unexpected behaviour. Each entry is "
        "tagged with the lifecycle phase it belonged to."
    ))
    for step in session.steps:
        cells.extend(_transcript_cells_for_step(step))

    nb["cells"] = cells
    nb["metadata"] = {
        "aiml_autopilot": {
            "project_id": project.id,
            "project_name": project.name,
            "session_id": session.session_id,
            "status": session.status,
            "lifecycle": PHASE_IDS,
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
