from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from aiml_discovery.ai_autopilot import AiAutopilot, AutopilotStep
from aiml_discovery.config import APP_NAME, PROJECT_HOME, UPLOAD_TYPES
from aiml_discovery.ingestion import load_dataset, list_sqlite_tables
from aiml_discovery.profiling import profile_dataframe
from aiml_discovery.reporting import build_markdown_report
from aiml_discovery.storage import DatasetInfo, ProjectInfo, ProjectStore
from aiml_discovery.training import TrainingSettings, train_automl

PAGES = [
    "Projects",
    "Data Sources",
    "Data Profile",
    "Training Lab",
    "Run History",
    "Model Report",
    "AI Autopilot",
]


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon=None, layout="wide")
    apply_style()

    store = ProjectStore()
    projects = store.list_projects()
    selected_project = render_sidebar(projects)

    st.title(APP_NAME)
    st.caption("Local model discovery for tabular business data")

    page = st.session_state.get("page", "Projects")
    if page == "Projects":
        render_projects(store, projects)
    elif selected_project is None:
        st.info("Create a project to begin.")
    elif page == "Data Sources":
        render_data_sources(store, selected_project)
    elif page == "Data Profile":
        render_data_profile(store, selected_project)
    elif page == "Training Lab":
        render_training_lab(store, selected_project)
    elif page == "Run History":
        render_run_history(store, selected_project)
    elif page == "Model Report":
        render_model_report(store, selected_project)
    elif page == "AI Autopilot":
        render_ai_autopilot(store, selected_project)


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; max-width: 1280px; }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e4e8ef;
            border-radius: 8px;
            padding: 14px 16px;
        }
        div[data-testid="stSidebar"] {
            border-right: 1px solid #e4e8ef;
        }
        .stButton > button {
            border-radius: 6px;
            border-color: #146c94;
        }
        .stDownloadButton > button {
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(projects: list[ProjectInfo]) -> ProjectInfo | None:
    st.sidebar.header("Workspace")
    project_lookup = {project.id: project for project in projects}
    project_ids = list(project_lookup.keys())

    if project_ids:
        current = st.session_state.get("project_id", project_ids[0])
        if current not in project_lookup:
            current = project_ids[0]
        selected_project_id = st.sidebar.selectbox(
            "Project",
            options=project_ids,
            index=project_ids.index(current),
            format_func=lambda project_id: project_lookup[project_id].name,
        )
        st.session_state["project_id"] = selected_project_id
    else:
        selected_project_id = None

    st.sidebar.radio("Workflow", PAGES, key="page")
    st.sidebar.divider()
    st.sidebar.caption(f"Project storage: {PROJECT_HOME}")
    return project_lookup.get(selected_project_id) if selected_project_id else None


def render_projects(store: ProjectStore, projects: list[ProjectInfo]) -> None:
    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        st.subheader("Create Project")
        with st.form("create_project_form", clear_on_submit=True):
            name = st.text_input("Project name", placeholder="Customer churn discovery")
            description = st.text_area("Description", height=100, placeholder="Business goal or model purpose")
            submitted = st.form_submit_button("Create project", type="primary")
        if submitted:
            try:
                project = store.create_project(name, description)
                st.session_state["project_id"] = project.id
                st.success(f"Created {project.name}.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with right:
        st.subheader("Projects")
        if not projects:
            st.info("No projects yet.")
            return

        table = pd.DataFrame(
            [
                {
                    "Name": project.name,
                    "Description": project.description,
                    "Updated": project.updated_at,
                    "ID": project.id,
                }
                for project in projects
            ]
        )
        st.dataframe(table, hide_index=True, use_container_width=True)


def render_data_sources(store: ProjectStore, project: ProjectInfo) -> None:
    st.subheader("Data Sources")
    uploaded = st.file_uploader("Add source", type=UPLOAD_TYPES)

    if uploaded is not None:
        if st.button("Add to project", type="primary"):
            try:
                saved_path = store.save_dataset_file(project.id, uploaded.name, uploaded.getvalue())
                registered = register_uploaded_source(store, project, saved_path, uploaded.name)
                if registered:
                    st.success(f"Added {len(registered)} dataset source(s).")
            except Exception as exc:
                st.error(str(exc))

    datasets = store.list_datasets(project.id)
    if not datasets:
        st.info("No data sources added yet.")
        return

    st.divider()
    st.subheader("Catalog")
    st.dataframe(dataset_table(datasets), hide_index=True, use_container_width=True)


def register_uploaded_source(
    store: ProjectStore,
    project: ProjectInfo,
    saved_path: Path,
    source_name: str,
) -> list[DatasetInfo]:
    suffix = saved_path.suffix.lower()
    datasets: list[DatasetInfo] = []

    if suffix in {".db", ".sqlite", ".sqlite3"}:
        for table_name in list_sqlite_tables(saved_path):
            loaded = load_dataset(saved_path, table_name=table_name)
            datasets.append(
                store.register_dataset(
                    project.id,
                    name=f"{saved_path.stem}.{table_name}",
                    source_name=source_name,
                    source_type=loaded.source_type,
                    file_path=saved_path,
                    table_name=table_name,
                    row_count=len(loaded.dataframe),
                    column_count=len(loaded.dataframe.columns),
                )
            )
        if not datasets:
            raise ValueError("No tables were found in the SQLite source.")
        return datasets

    loaded = load_dataset(saved_path)
    datasets.append(
        store.register_dataset(
            project.id,
            name=loaded.name,
            source_name=source_name,
            source_type=loaded.source_type,
            file_path=saved_path,
            table_name=loaded.table_name,
            row_count=len(loaded.dataframe),
            column_count=len(loaded.dataframe.columns),
        )
    )
    return datasets


def render_data_profile(store: ProjectStore, project: ProjectInfo) -> None:
    st.subheader("Data Profile")
    dataset = choose_dataset(store, project, "profile_dataset")
    if dataset is None:
        return

    try:
        loaded = load_dataset(dataset.file_path, table_name=dataset.table_name)
        profile = profile_dataframe(loaded.dataframe)
    except Exception as exc:
        st.error(str(exc))
        return

    metric_row(
        [
            ("Rows", profile["row_count"]),
            ("Columns", profile["column_count"]),
            ("Missing", f"{profile['missing_pct']:.1f}%"),
            ("Duplicates", profile["duplicate_rows"]),
        ]
    )

    st.subheader("Preview")
    st.dataframe(loaded.dataframe.head(100), use_container_width=True)

    columns = pd.DataFrame(profile["columns"])
    st.subheader("Column Health")
    st.dataframe(columns, hide_index=True, use_container_width=True)

    missing = columns[columns["missing_pct"] > 0].copy()
    if not missing.empty:
        st.plotly_chart(
            px.bar(missing, x="name", y="missing_pct", color="role", title="Missing Values"),
            use_container_width=True,
        )

    if profile["numeric_summary"]:
        st.subheader("Numeric Summary")
        st.dataframe(pd.DataFrame(profile["numeric_summary"]), hide_index=True, use_container_width=True)


def render_training_lab(store: ProjectStore, project: ProjectInfo) -> None:
    st.subheader("Training Lab")
    dataset = choose_dataset(store, project, "training_dataset")
    if dataset is None:
        return

    try:
        loaded = load_dataset(dataset.file_path, table_name=dataset.table_name)
    except Exception as exc:
        st.error(str(exc))
        return

    target_column = st.selectbox("Target", loaded.dataframe.columns)
    control_a, control_b, control_c = st.columns(3)
    with control_a:
        test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    with control_b:
        random_state = st.number_input("Random seed", min_value=0, value=42, step=1)
    with control_c:
        max_rows = st.number_input("Max rows", min_value=0, value=0, step=500)

    if st.button("Train models", type="primary"):
        settings = TrainingSettings(
            target_column=str(target_column),
            test_size=float(test_size),
            random_state=int(random_state),
            max_rows=int(max_rows) if max_rows else None,
        )
        try:
            with st.spinner("Training candidate models"):
                result, model = train_automl(loaded.dataframe, settings)
                profile = profile_dataframe(loaded.dataframe)
                metadata = result.to_metadata()
                metadata["dataset"] = dataset.to_dict()
                report = build_markdown_report(project.name, dataset.to_dict(), metadata, profile)
                run_path = store.save_run(project.id, metadata, model, report)
            st.success(f"Saved run {result.run_id}.")
            st.session_state["latest_run_path"] = str(run_path)
            render_training_result(metadata)
        except Exception as exc:
            st.error(str(exc))


def render_run_history(store: ProjectStore, project: ProjectInfo) -> None:
    st.subheader("Run History")
    runs = store.list_runs(project.id)
    if not runs:
        st.info("No saved model runs yet.")
        return

    st.dataframe(runs_table(runs), hide_index=True, use_container_width=True)

    selected_run = choose_run(runs, "history_run")
    if selected_run:
        render_training_result(selected_run)
        report_path = selected_run.get("report_path")
        if report_path:
            report = store.read_report(report_path)
            st.download_button(
                "Download report",
                data=report,
                file_name=f"{selected_run.get('run_id', 'model_report')}.md",
                mime="text/markdown",
            )


def render_model_report(store: ProjectStore, project: ProjectInfo) -> None:
    st.subheader("Model Report")
    runs = store.list_runs(project.id)
    if not runs:
        st.info("No model reports yet.")
        return

    selected_run = choose_run(runs, "report_run")
    if not selected_run:
        return

    report_path = selected_run.get("report_path")
    if not report_path:
        st.error("This run does not have a saved report.")
        return

    report = store.read_report(report_path)
    st.markdown(report)
    st.download_button(
        "Download report",
        data=report,
        file_name=f"{selected_run.get('run_id', 'model_report')}.md",
        mime="text/markdown",
    )


def choose_dataset(store: ProjectStore, project: ProjectInfo, key: str) -> DatasetInfo | None:
    datasets = store.list_datasets(project.id)
    if not datasets:
        st.info("Add a data source first.")
        return None

    dataset_lookup = {dataset.id: dataset for dataset in datasets}
    dataset_id = st.selectbox(
        "Dataset",
        options=list(dataset_lookup.keys()),
        format_func=lambda value: dataset_lookup[value].name,
        key=key,
    )
    return dataset_lookup[dataset_id]


def choose_run(runs: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    run_lookup = {run["run_id"]: run for run in runs}
    run_id = st.selectbox(
        "Run",
        options=list(run_lookup.keys()),
        format_func=lambda value: f"{value} - {run_lookup[value].get('best_model_name', '')}",
        key=key,
    )
    return run_lookup.get(run_id)


def dataset_table(datasets: list[DatasetInfo]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Name": dataset.name,
                "Type": dataset.source_type,
                "Rows": dataset.row_count,
                "Columns": dataset.column_count,
                "Uploaded": dataset.uploaded_at,
                "Source": dataset.source_name,
            }
            for dataset in datasets
        ]
    )


def runs_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        dataset = run.get("dataset", {})
        metrics = run.get("best_metrics", {})
        rows.append(
            {
                "Run": run.get("run_id"),
                "Dataset": dataset.get("name", ""),
                "Task": run.get("task_type"),
                "Target": run.get("target_column"),
                "Best model": run.get("best_model_name"),
                "Primary score": primary_score(run.get("task_type"), metrics),
                "Saved": run.get("saved_at"),
            }
        )
    return pd.DataFrame(rows)


def primary_score(task_type: str | None, metrics: dict[str, Any]) -> str:
    if task_type == "classification":
        return f"F1 {metrics.get('f1_weighted', 0):.3f}"
    if task_type == "regression":
        return f"R2 {metrics.get('r2', 0):.3f}"
    return ""


def render_training_result(metadata: dict[str, Any]) -> None:
    metrics = metadata.get("best_metrics", {})
    metric_row(
        [
            ("Task", metadata.get("task_type", "")),
            ("Best model", metadata.get("best_model_name", "")),
            ("Target", metadata.get("target_column", "")),
            ("Rows", metadata.get("row_count", "")),
        ]
    )

    if metrics:
        st.subheader("Best Metrics")
        metric_row([(key, _metric_value(value)) for key, value in metrics.items()])

    leaderboard = leaderboard_table(metadata.get("leaderboard", []))
    if not leaderboard.empty:
        st.subheader("Leaderboard")
        st.dataframe(leaderboard, hide_index=True, use_container_width=True)


def leaderboard_table(leaderboard: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in leaderboard:
        row = {
            "Rank": entry.get("rank"),
            "Model": entry.get("model"),
            "Status": entry.get("status"),
            "Error": entry.get("error"),
        }
        for metric, value in entry.get("metrics", {}).items():
            row[metric] = round(value, 4) if isinstance(value, float) else value
        rows.append(row)
    return pd.DataFrame(rows)


def metric_row(items: list[tuple[str, Any]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, _metric_value(value))


def _metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_ai_autopilot(store: ProjectStore, project: ProjectInfo) -> None:
    st.subheader("AI Autopilot")
    st.caption("Let GPT-4o analyse your data, devise a strategy, and execute training automatically.")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        st.success("OpenAI API key loaded from environment variable.")
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Set OPENAI_API_KEY env var to avoid entering it here.",
        )

    datasets = store.list_datasets(project.id)
    if not datasets:
        st.info("No data sources found. Upload at least one dataset in 'Data Sources' first.")
        return

    st.info(f"{len(datasets)} dataset(s) available for analysis.")

    if not st.button("Launch AI Analysis", type="primary", disabled=not api_key):
        return

    autopilot = AiAutopilot(api_key, project.id, store)

    try:
        with st.status("AI is analysing your data...", expanded=True) as status:
            for step in autopilot.run():
                _render_autopilot_step(step)
            status.update(label="Analysis complete!", state="complete", expanded=True)
    except Exception as exc:
        st.error(f"AI Autopilot error: {exc}")
        return

    if autopilot.new_datasets:
        st.subheader("New Data Sources Created")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Name": ds.name,
                        "Rows": ds.row_count,
                        "Columns": ds.column_count,
                        "Type": ds.source_type,
                    }
                    for ds in autopilot.new_datasets
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    if autopilot.training_runs:
        st.subheader("Training Results")
        rows = []
        for r in autopilot.training_runs:
            row: dict[str, Any] = {
                "Dataset": r["dataset"],
                "Target": r["target"],
                "Task": r["task_type"],
                "Best Model": r["best_model"],
            }
            row.update(
                {k: round(v, 4) for k, v in r.get("best_metrics", {}).items()}
            )
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if autopilot.strategy_summary:
        st.subheader("Final Strategy Report")
        st.markdown(autopilot.strategy_summary)


def _render_autopilot_step(step: AutopilotStep) -> None:
    if step.kind == "thought":
        with st.expander(f"Step {step.index}: AI Reasoning", expanded=False):
            st.markdown(step.detail)
    elif step.kind == "tool_call":
        st.write(f"**Step {step.index}:** ⚙ {step.title}")
    elif step.kind == "tool_result":
        st.success(f"Step {step.index}: {step.title} — {step.detail}")
    elif step.kind == "chart":
        with st.expander(f"Step {step.index}: {step.title}", expanded=True):
            if step.detail:
                st.caption(step.detail)
            if step.data and "figure" in step.data:
                st.plotly_chart(
                    step.data["figure"],
                    use_container_width=True,
                    key=f"autopilot_chart_{step.index}",
                )
    elif step.kind == "new_dataset":
        st.success(f"**Step {step.index}: {step.title}** — {step.detail}")
    elif step.kind == "training":
        with st.expander(f"Step {step.index}: {step.title}", expanded=True):
            st.write(step.detail)
    elif step.kind == "summary":
        with st.expander(f"Step {step.index}: Final Strategy", expanded=True):
            st.markdown(step.detail)


if __name__ == "__main__":
    main()

