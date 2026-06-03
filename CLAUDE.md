# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend (Python/FastAPI)

```powershell
# Setup (one-time)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Run API server (primary backend)
uvicorn aiml_discovery.api:app --reload

# Run Streamlit UI (alternative interface)
streamlit run app.py

# Run tests
pytest

# Run a single test file
pytest tests/test_training.py -v
```

### Frontend (React/Vite)

```powershell
cd react-frontend
npm.cmd install   # Windows; use npm install on other platforms
npm.cmd run dev   # Dev server at http://127.0.0.1:5173
npm.cmd run build
```

### Full Stack Development

Run API in one terminal (`uvicorn aiml_discovery.api:app --reload`), React in another (`cd react-frontend && npm.cmd run dev`). The Vite dev server proxies `/api` to `http://127.0.0.1:8082`, so no CORS config is needed in dev.

### Environment

Copy `.env` with your `OPENAI_API_KEY`. For Azure OpenAI, also set `OPENAI_API_BASE`. Sessions and project data persist to `~/.aiml_discovery/` (override with `AIML_DISCOVERY_HOME`).

## Architecture

This is a local-first tabular ML discovery platform with three user interfaces sharing one Python backend:

### Three Interfaces

1. **React Autopilot Dashboard** (`react-frontend/`) — The primary modern UI. Dark "Tech Academy" theme, Vite + React 18 + Plotly.js. Shows a swimlane graph or linear timeline of agent steps as they happen via SSE. Keyboard shortcuts: `G` (graph view), `T` (timeline view). Slide-in detail drawer, tweaks panel, notebook download.

2. **Streamlit Workspace** (`app.py`) — Legacy manual interface. Pages for project creation, dataset upload, data profiling, AutoML training, leaderboard comparison, and an "AI Autopilot" page that embeds the agent workflow.

3. **FastAPI HTTP Service** (`aiml_discovery/api.py`) — REST + Server-Sent Events. Used by the React frontend; also consumable directly. Docs at `http://127.0.0.1:8082/docs`.

### Agent System

The core AI engine is a multi-agent orchestrator that runs CRISP-DM autonomously:

- **`AimlScientist`** (`aiml_discovery/agents/scientist.py`) — Orchestrator LLM. Loops up to 60 iterations deciding which specialist to invoke next. Can pause and emit `ask_user` events.
- **`EdaAgent`** — Data profiling and Plotly chart generation; vision-capable (passes chart images back to the LLM).
- **`FeatureEngineeringAgent`** — 50+ dataset transformations.
- **`ModelingAgent`** — AutoML baseline across 25+ scikit-learn models.
- **`ReviewAgent`** — Critiques for leakage, imbalance, underfitting.
- **`FineTuningAgent`** — Iterates on improvements after review.
- **`ResearcherAgent`** — Background domain research.
- **`DriftAgent`** — Feature/label drift detection.

All agents extend `BaseAgent` (`aiml_discovery/agents/base.py`) and yield `AutopilotStep` dataclass instances. The orchestrator thin-façade is `AiAutopilot` (`aiml_discovery/ai_autopilot.py`).

### Session Model

Each autopilot run is a **session** with a background worker thread. Steps are persisted to disk and streamed to the frontend via SSE. Sessions can be resumed after a restart. The full flow:

```
POST /api/projects/{pid}/autopilot/sessions        → create session, get session_id
GET  /api/projects/{pid}/autopilot/sessions/{sid}/events   → SSE stream of AutopilotStep
POST /api/projects/{pid}/autopilot/sessions/{sid}/answers  → submit user reply (resumes paused run)
GET  /api/projects/{pid}/autopilot/sessions/{sid}/notebook → download as .ipynb
```

### Notebook Export

Every agent step produces a Jupyter cell with runnable code (Plotly Express, scikit-learn), a markdown explanation, and actual outputs (charts as base64 PNG, DataFrames as CSV). Exported notebooks are self-contained and human-readable.

### Key Modules

| File | Purpose |
|------|---------|
| `aiml_discovery/api.py` | All HTTP routes, session threading, SSE streaming |
| `aiml_discovery/agents/scientist.py` | LLM orchestrator loop |
| `aiml_discovery/agents/base.py` | `AgentContext`, `AutopilotStep`, `BaseAgent` |
| `aiml_discovery/ingestion.py` | Dataset loading (CSV, Excel, JSON, SQLite) |
| `aiml_discovery/training.py` | AutoML training with optional XGBoost/LGB/CatBoost |
| `aiml_discovery/notebook_export.py` | Jupyter notebook generation |
| `aiml_discovery/session_store.py` | Session persistence |
| `aiml_discovery/config.py` | `APP_NAME`, `PROJECT_HOME`, supported file extensions |
| `react-frontend/vite.config.js` | Port 5173, `/api` proxy to port 8082 |

### Optional Dependencies

XGBoost, LightGBM, CatBoost (AutoML models), SHAP (explainability), Optuna (Bayesian HPO), and pmdarima (time-series) are gracefully skipped if not installed.

## Documentation

- `docs/AUTOPILOT.md` — Agent roles and CRISP-DM lifecycle
- `docs/AUTOPILOT_API.md` — Full HTTP endpoint reference with curl examples
- `docs/AGENT_PLAYBOOK.md` — Agent internal decision-making and tool signatures
- `docs/AGENT_TOOLS.md` — Tool definitions agents can invoke
- `AGENT_ROADMAP.md` — Capability checklist
