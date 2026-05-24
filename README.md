# AIML Discovery Training UI

A local-first tabular ML discovery workspace. You can drive it the manual way
through the Streamlit app (upload data, profile, train, compare), or hand a
goal to the **AI Autopilot** — a multi-agent system that runs the full
CRISP-DM lifecycle for you and exports a runnable Jupyter handover notebook
at the end.

The recommended interface is the React Autopilot Dashboard (live agent run
viewer, dark "Tech Academy" UI). The Streamlit app and the FastAPI server
remain available for manual workflows and programmatic access.

## What's in the box

- **AI Autopilot** — orchestrated agents (Scientist, Researcher, EDA, Feature
  Engineering, Modeling, Review, Fine Tuning) collaborate to take a dataset
  from EDA → features → AutoML → review → tuning, asking for clarification
  only when needed.
- **React Autopilot Dashboard** — swimlane graph and linear timeline views of
  the agent run, with a slide-in detail drawer, live event stream, tweaks
  panel, and `G` / `T` keyboard shortcuts. Lives in `react-frontend/`.
- **Streamlit manual workspace** — project creation, dataset upload (CSV,
  Excel, JSON, SQLite), data profiling, AutoML training, leaderboard
  comparison, saved runs with artifacts and reports. Lives in `app.py`.
- **FastAPI service** — REST + Server-Sent Events for projects, datasets,
  autopilot sessions, follow-ups, and notebook export. Lives in
  `aiml_discovery/api.py`.
- **Notebook export** — every session can be downloaded as a structured
  Jupyter notebook organised by lifecycle phase, with readable
  `plotly.express` code for each chart (no JSON-spec dumps) and a clean
  appendix transcript (tool-call noise suppressed).

## Setup

Use Python 3.12 or 3.13 — some dependencies don't yet build cleanly under
3.14.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set `OPENAI_API_KEY` (and optionally Azure OpenAI variables) in a `.env`
file or your shell before starting the API — the agents won't run without
LLM credentials. The frontend never asks for keys; configure them on the
backend only.

## Run the AI Autopilot (recommended)

Start the FastAPI server and the React dashboard in two terminals:

```powershell
# terminal 1 — backend
uvicorn aiml_discovery.api:app --reload
```

```powershell
# terminal 2 — frontend
cd react-frontend
npm.cmd install
npm.cmd run dev
```

Open:

```text
http://127.0.0.1:5173
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`, so no CORS
config is needed. To point at a different backend, set `VITE_API_BASE`
before running Vite.

Inside the dashboard:

- Pick a project, upload or select a dataset, type a goal, and hit **launch
  run**.
- The swimlane graph shows each agent in its own lane with cards positioned
  by handoff order; the viewport auto-scrolls to keep the running step in
  view. Click any card (or any timeline row) to open the detail drawer.
- Toggle between **Graph** and **Timeline** with the segmented control or
  `G` / `T`. Use the floating **tweaks** panel for density, replay speed,
  legend, and the live stream footer.
- When the run finishes (or pauses for input), download the handover
  notebook from the header link.

## Run the Streamlit manual workspace

```powershell
streamlit run app.py
```

## Run the API only

```powershell
uvicorn aiml_discovery.api:app --reload
```

OpenAPI docs:

```text
http://127.0.0.1:8000/docs
```

Common AI Autopilot endpoints (full reference in [docs/AUTOPILOT_API.md](docs/AUTOPILOT_API.md)):

- `POST /api/projects/{project_id}/autopilot/sessions` — starts a run.
- `GET /api/projects/{project_id}/autopilot/sessions/{session_id}/events` —
  Server-Sent Events stream until the run pauses or completes.
- `POST /api/projects/{project_id}/autopilot/sessions/{session_id}/answers` —
  submits answers when the run is waiting for input.
- `POST /api/projects/{project_id}/autopilot/sessions/{session_id}/messages` —
  sends a follow-up to a completed session.
- `GET /api/projects/{project_id}/autopilot/sessions/{session_id}/notebook` —
  downloads the session notebook as `.ipynb`.

## The handover notebook

Each completed (or interrupted) session can be exported as a Jupyter
notebook structured by CRISP-DM phase:

- Business Understanding → Data Understanding → Data Preparation → Modeling
  → Evaluation → Iteration.
- Every chart is emitted as **runnable Plotly Express code** that re-creates
  the figure from the loaded `df_<dataset>` variable — no opaque JSON specs.
- Every derived dataset shows the FE operation, rationale, and a
  `pd.read_csv(...)` of the materialised CSV.
- Every training run includes a `joblib.load(...)` of the saved model and
  recreates the exact train/test split the Modeling Agent used.
- The appendix carries the chronological agent transcript (reasoning,
  observations, charts, Q&A, training, review, summary). Tool-call /
  tool-result / agent-start / agent-end entries are suppressed because
  they're noise for a human reader.

## Project data

By default, project artifacts live outside the repository at:

```text
~/.aiml_discovery/projects
```

Set `AIML_DISCOVERY_HOME` to a different local folder if you want.

## Test

```powershell
pytest
```

## Further reading

- [docs/AUTOPILOT.md](docs/AUTOPILOT.md) — agent roles, lifecycle phases,
  and how the orchestration works.
- [docs/AUTOPILOT_API.md](docs/AUTOPILOT_API.md) — full HTTP / SSE API
  reference.
- [react-frontend/README.md](react-frontend/README.md) — frontend dev notes.
