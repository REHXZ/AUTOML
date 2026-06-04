import io
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, abort, jsonify, request

from backend.logic.notebook_export import build_notebook, serialize_notebook
from backend.server.helpers import load_session_or_404, project_or_404
from backend.services.project_store import ProjectStore

runs_bp = Blueprint("runs", __name__)


@runs_bp.get("/api/projects/<project_id>/runs")
def list_runs_api(project_id: str):
    store = ProjectStore()
    project_or_404(store, project_id)
    runs = store.list_runs(project_id)
    safe = [
        {k: v for k, v in r.items() if k not in {"diagnostics", "leaderboard"}}
        for r in runs
    ]
    return jsonify({"runs": safe})


@runs_bp.get("/api/projects/<project_id>/runs/<run_id>")
def get_run_api(project_id: str, run_id: str):
    store = ProjectStore()
    project_or_404(store, project_id)
    runs = store.list_runs(project_id)
    run = next((r for r in runs if r.get("run_id") == run_id), None)
    if run is None:
        abort(404, description=f"Run '{run_id}' not found in project '{project_id}'.")
    return jsonify(run)


@runs_bp.get("/api/projects/<project_id>/runs/<run_id>/charts")
def get_run_charts_api(project_id: str, run_id: str):
    from backend.logic.diagnostics import (
        build_diagnostic_figures,
        build_feature_importance_figure,
        build_leaderboard_figure,
        build_residuals_over_time_figure,
    )

    store = ProjectStore()
    project_or_404(store, project_id)
    runs = store.list_runs(project_id)
    run = next((r for r in runs if r.get("run_id") == run_id), None)
    if run is None:
        abort(404, description=f"Run '{run_id}' not found in project '{project_id}'.")

    diagnostics = run.get("diagnostics") or {}
    leaderboard = run.get("leaderboard") or []
    task_type = run.get("task_type", "")
    target_column = run.get("target_column", "")
    model_path = run.get("model_path", "")
    best_model_name = run.get("best_model_name", "")

    charts = []

    for title, fig in build_diagnostic_figures(diagnostics, target_column):
        charts.append({"title": title, "figure_json": fig.to_json()})

    rot_fig = build_residuals_over_time_figure(diagnostics, target_column)
    if rot_fig is not None:
        charts.append({"title": "Residuals over Time", "figure_json": rot_fig.to_json()})

    if model_path and Path(model_path).exists():
        fi_fig = build_feature_importance_figure(model_path, run_label=best_model_name)
        if fi_fig is not None:
            charts.append({"title": "Feature Importance", "figure_json": fi_fig.to_json()})

    lb_fig = build_leaderboard_figure(leaderboard, task_type, run_label=run_id)
    if lb_fig is not None:
        charts.append({"title": "Leaderboard", "figure_json": lb_fig.to_json()})

    return jsonify({"run_id": run_id, "charts": charts})


@runs_bp.post("/api/projects/<project_id>/runs/<run_id>/score")
def score_run_api(project_id: str, run_id: str):
    import joblib
    import pandas as pd

    store = ProjectStore()
    project_or_404(store, project_id)

    runs = store.list_runs(project_id)
    run = next((r for r in runs if r.get("run_id") == run_id), None)
    if run is None:
        abort(404, description=f"Run '{run_id}' not found in project '{project_id}'.")

    model_path = run.get("model_path")
    if not model_path or not Path(model_path).exists():
        abort(404, description=f"Model file not found for run '{run_id}'. Was it saved?")

    target_column = run.get("target_column", "")
    task_type = run.get("task_type", "")

    content_type = request.content_type or ""
    if "multipart/form-data" in content_type:
        file = request.files.get("file")
        if file is None:
            abort(400, description="Multipart request must include a 'file' field.")
        try:
            df = pd.read_csv(io.BytesIO(file.read()))
        except Exception as exc:
            abort(400, description=f"Could not parse uploaded CSV: {exc}")
    else:
        body = request.get_json(silent=True) or {}
        dataset_id = body.get("dataset_id")
        if dataset_id:
            from backend.logic.ingestion import load_dataset as _load_dataset
            datasets = store.list_datasets(project_id)
            ds = next((d for d in datasets if d.id == dataset_id), None)
            if ds is None:
                abort(404, description=f"Dataset '{dataset_id}' not found in project '{project_id}'.")
            try:
                df = _load_dataset(ds.file_path, ds.table_name).dataframe
            except Exception as exc:
                abort(400, description=f"Could not read dataset '{dataset_id}': {exc}")
        else:
            rows = body.get("data")
            if not isinstance(rows, list) or not rows:
                abort(400, description="JSON body must contain either 'dataset_id' or a non-empty 'data' array.")
            try:
                df = pd.DataFrame(rows)
            except Exception as exc:
                abort(400, description=f"Could not build DataFrame from data: {exc}")

    feature_df = df.drop(columns=[target_column], errors="ignore")
    if feature_df.empty or len(feature_df.columns) == 0:
        abort(400, description="No feature columns found in the input data.")

    try:
        pipeline = joblib.load(model_path)
        predictions = pipeline.predict(feature_df)
    except Exception as exc:
        abort(500, description=f"Scoring failed: {exc}")

    preds_list = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)

    result: dict[str, Any] = {
        "run_id": run_id,
        "task_type": task_type,
        "predictions": preds_list,
        "n_rows": len(preds_list),
    }

    if task_type == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(feature_df)
            result["probabilities"] = proba.tolist()
        except Exception:
            pass

    return jsonify(result)


@runs_bp.get("/api/projects/<project_id>/autopilot/sessions/<session_id>/notebook")
def download_autopilot_notebook_api(project_id: str, session_id: str):
    store = ProjectStore()
    project = project_or_404(store, project_id)
    loaded = load_session_or_404(store, project_id, session_id)
    notebook = build_notebook(project, loaded, store)
    data = serialize_notebook(notebook)
    filename = f"{project_id}_{session_id}.ipynb"
    return Response(
        data,
        mimetype="application/x-ipynb+json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
