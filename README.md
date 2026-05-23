# AIML Discovery Training UI

A local-first Streamlit application for no-code tabular ML discovery. Employees can create projects, upload data sources, profile datasets, train supervised models, compare candidate models, and save model runs with reports.

## Features

- Project workspace creation with local artifact storage.
- Data ingestion for CSV, Excel, JSON, and SQLite files.
- Data profiling with column health, missingness, duplicates, and numeric summaries.
- AutoML-style training for tabular classification and regression.
- Leaderboard comparison across baseline scikit-learn models.
- Saved run history with model artifacts and downloadable markdown reports.

## Setup

Use Python 3.12 or 3.13 if your local Python 3.14 environment has package compatibility issues.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

## Run the API

```powershell
uvicorn aiml_discovery.api:app --reload
```

The OpenAPI docs are available at:

```text
http://127.0.0.1:8000/docs
```

Common AI Autopilot endpoints:

- `POST /api/projects/{project_id}/autopilot/sessions` starts a run.
- `GET /api/projects/{project_id}/autopilot/sessions/{session_id}/events` streams Server-Sent Events until the run pauses or completes.
- `POST /api/projects/{project_id}/autopilot/sessions/{session_id}/answers` submits answers when the run is waiting for input.
- `POST /api/projects/{project_id}/autopilot/sessions/{session_id}/messages` sends follow-up work to a completed session.
- `GET /api/projects/{project_id}/autopilot/sessions/{session_id}/notebook` downloads the session notebook.

By default, project data is stored outside the repository at:

```text
~/.aiml_discovery/projects
```

Set `AIML_DISCOVERY_HOME` to use another local folder.

## Test

```powershell
pytest
```
