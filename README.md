# AIML Discovery Training UI

[![License: AGPL v3](https://img.shields.io/badge/Community-AGPL--v3-blue.svg)](LICENSE)
[![License: Commercial](https://img.shields.io/badge/Enterprise-Commercial-orange.svg)](LICENSE)

A local-first tabular ML discovery workspace. Hand a goal to the **AI Autopilot** — a multi-agent system that runs the full CRISP-DM lifecycle for you and exports a runnable Jupyter handover notebook at the end.

The recommended interface is the React Autopilot Dashboard (live agent run viewer, dark "Tech Academy" UI).

## What's in the box

- **AI Autopilot** — orchestrated agents (Scientist, Researcher, EDA, Feature Engineering, Modeling, Review, Fine Tuning) collaborate to take a dataset from EDA → features → AutoML → review → tuning, asking for clarification only when needed.
- **React Autopilot Dashboard** — swimlane graph and linear timeline views of the agent run, with a slide-in detail drawer, live event stream, tweaks panel, and `G` / `T` keyboard shortcuts. Lives in `react-frontend/`.
- **Flask HTTP Service** — REST + Server-Sent Events for projects, datasets, autopilot sessions, follow-ups, and notebook export. Lives in `backend/server/app.py`.
- **Notebook export** — every session can be downloaded as a structured Jupyter notebook organised by lifecycle phase, with readable `plotly.express` code for each chart and a clean appendix transcript.

## Requirements

- **Python** 3.10+ — some dependencies don't yet build cleanly under 3.14.
- **Node.js** 18+ with npm — required for the React frontend.
- An **OpenAI API key** (or Azure OpenAI credentials) — the agents won't run without LLM credentials.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/rehxz/aiml-gui.git
cd aiml-gui
```

### 2. Create and activate a Python virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional heavy dependencies (gracefully skipped if absent):

| Package | Purpose |
|---------|---------|
| `xgboost` | XGBoost AutoML model |
| `lightgbm` | LightGBM AutoML model |
| `catboost` | CatBoost AutoML model |
| `shap` | Feature importance / explainability |
| `optuna` | Bayesian hyperparameter optimisation |
| `pmdarima` | Time-series support |
| `statsmodels` | VIF, stationarity tests, seasonal decompose |
| `imbalanced-learn` | SMOTE and resampling operations |

Install any subset with `pip install xgboost lightgbm catboost shap optuna pmdarima statsmodels imbalanced-learn`.

### 4. Configure environment variables

Create a `.env` file in the project root (never commit this file):

```dotenv
# Required — OpenAI
OPENAI_API_KEY=sk-...

# Optional — Azure OpenAI (leave out if using standard OpenAI)
OPENAI_API_BASE=https://<your-resource>.openai.azure.com/
OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Optional — change where project artifacts are stored (default: ~/.aiml_discovery)
AIML_DISCOVERY_HOME=/path/to/your/projects
```

The frontend never handles API keys — configure them on the backend only.

### 5. Install frontend dependencies

```bash
cd react-frontend
npm install
cd ..
```

> **Windows note:** if `npm` is not on your PATH, use `npm.cmd` instead.

## Starting the server

### Full stack (recommended)

Start the Flask backend and the React dashboard in two separate terminals:

**Terminal 1 — backend**
```bash
flask --app backend.server.app run --port 8082 --debug
```

**Terminal 2 — frontend**
```bash
cd react-frontend
npm run dev
```

> **Windows PowerShell:** use `npm.cmd run dev`.

Open **http://127.0.0.1:5173** in your browser.

The Vite dev server proxies `/api` to `http://127.0.0.1:8082`, so no CORS config is needed in dev. To point at a different backend, set `VITE_API_BASE` before running Vite.

### Backend only

```bash
flask --app backend.server.app run --port 8082 --debug
```

API health check: **http://127.0.0.1:8082/api/health**

Common AI Autopilot endpoints (full reference in [docs/AUTOPILOT_API.md](docs/AUTOPILOT_API.md)):

- `POST /api/projects/{project_id}/autopilot/sessions` — starts a run.
- `GET /api/projects/{project_id}/autopilot/sessions/{session_id}/events` — Server-Sent Events stream until the run pauses or completes.
- `POST /api/projects/{project_id}/autopilot/sessions/{session_id}/answers` — submits answers when the run is waiting for input.
- `POST /api/projects/{project_id}/autopilot/sessions/{session_id}/messages` — sends a follow-up to a completed session.
- `GET /api/projects/{project_id}/autopilot/sessions/{session_id}/notebook` — downloads the session notebook as `.ipynb`.

## Using the dashboard

- Pick a project, upload or select a dataset, type a goal, and hit **Launch Run**.
- The swimlane graph shows each agent in its own lane with cards positioned by handoff order; the viewport auto-scrolls to keep the running step in view. Click any card (or any timeline row) to open the detail drawer.
- Toggle between **Graph** and **Timeline** with the segmented control or `G` / `T`. Use the floating **tweaks** panel for density, replay speed, legend, and the live stream footer.
- When the run finishes (or pauses for input), download the handover notebook from the header link.

## Backend structure

```
backend/
├── config/settings.py          # APP_NAME, PROJECT_HOME, supported extensions
├── logic/
│   ├── autopilot.py            # AiAutopilot facade
│   ├── training.py             # AutoML training (25+ models)
│   ├── ingestion.py            # Dataset loading (CSV/Excel/JSON/SQLite)
│   ├── notebook_export.py      # Jupyter .ipynb generation
│   └── agents/                 # EDA, FE, Modeling, Review, FineTuning, Drift...
├── server/
│   ├── app.py                  # Flask app + blueprint registration
│   ├── job_manager.py          # Background worker threads
│   ├── streaming.py            # SSE helper
│   └── routes/                 # health, projects, datasets, sessions, runs
└── services/
    ├── project_store.py        # ProjectStore, ProjectInfo, DatasetInfo
    ├── session_store.py        # Session persistence
    └── tracing.py              # Optional Phoenix/OpenTelemetry tracing
```

## The handover notebook

Each completed (or interrupted) session can be exported as a Jupyter notebook structured by CRISP-DM phase:

- Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Iteration.
- Every chart is emitted as **runnable Plotly Express code** that re-creates the figure from the loaded `df_<dataset>` variable.
- Every derived dataset shows the FE operation, rationale, and a `pd.read_csv(...)` of the materialised CSV.
- Every training run includes a `joblib.load(...)` of the saved model and recreates the exact train/test split.
- The appendix carries the chronological agent transcript with tool-call noise suppressed.

## Project data

By default, project artifacts live outside the repository at:

```
~/.aiml_discovery/projects
```

Set `AIML_DISCOVERY_HOME` to a different local folder if you want.

## Tests

```bash
pytest
# or run a single file:
pytest tests/test_training.py -v
```

## Further reading

- [docs/AUTOPILOT.md](docs/AUTOPILOT.md) — agent roles, lifecycle phases, and how the orchestration works.
- [docs/AUTOPILOT_API.md](docs/AUTOPILOT_API.md) — full HTTP / SSE API reference.
- [react-frontend/README.md](react-frontend/README.md) — frontend dev notes.

## License

This project uses a **dual license** model:

| Use case | License | Cost |
|----------|---------|------|
| Open-source / community projects | [GNU AGPL v3.0](LICENSE) | Free |
| Proprietary / enterprise / SaaS products | [Commercial License](LICENSE) | Paid |

**Community (AGPL v3.0):** Free to use, modify, and distribute. Any software that incorporates or is derived from this project — including network-facing deployments — must also be released under the AGPL v3.0 with full source code made publicly available.

**Enterprise / Commercial:** If you want to use this software in a proprietary product or service without the AGPL's source-disclosure requirements, you need a Commercial License. Contact **rehxxz@gmail.com** to obtain one.
