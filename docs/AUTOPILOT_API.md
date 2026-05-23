# AI Autopilot API

The AI Autopilot API exposes the existing multi-agent Autopilot engine over HTTP.
It uses the same local project store, datasets, model runs, and persisted session
files as the Streamlit UI.

## Start the API Server

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the local API:

```powershell
uvicorn aiml_discovery.api:app --reload
```

Open the generated API docs:

```text
http://127.0.0.1:8000/docs
```

The API key can come from `.env`:

```text
OPENAI_API_KEY=your_key_here
```

Or it can be passed in the JSON body when starting or continuing a session.

## How It Works

Autopilot runs are session-based because the underlying workflow can take time
and may pause to ask the user questions.

The normal lifecycle is:

1. Create or use an existing project.
2. Register at least one dataset.
3. Start an Autopilot session.
4. Read session status or stream step events.
5. If the session pauses, submit answers.
6. When complete, download the notebook or send follow-up instructions.

Status values:

| Status | Meaning |
|---|---|
| `running` | The background Autopilot worker is active. |
| `waiting_for_input` | Autopilot asked questions and needs answers. |
| `complete` | The run finished and produced final outputs. |
| `idle` | A saved session is available but not running. |
| `error` | The worker stopped because of an exception. |

## Endpoint Summary

### Health

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Check that the API server is running. |

### Projects

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/projects` | List projects. |
| `POST` | `/api/projects` | Create a project. |
| `GET` | `/api/projects/{project_id}` | Get one project. |

Create a project:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/projects" `
  -ContentType "application/json" `
  -Body '{"name":"Customer Churn","description":"Predict churn from tabular data"}'
```

### Datasets

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/projects/{project_id}/datasets` | List registered datasets. |
| `POST` | `/api/projects/{project_id}/datasets/register` | Register a local CSV, Excel, JSON, or SQLite file. |

Register a dataset:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/projects/customer_churn/datasets/register" `
  -ContentType "application/json" `
  -Body '{"file_path":"C:\\data\\customers.csv"}'
```

For SQLite files, omit `table_name` to register every table, or pass `table_name`
to register a specific table.

### Autopilot Sessions

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/projects/{project_id}/autopilot/sessions` | List saved Autopilot sessions. |
| `POST` | `/api/projects/{project_id}/autopilot/sessions` | Start a new Autopilot session. |
| `GET` | `/api/projects/{project_id}/autopilot/sessions/{session_id}` | Read session state and outputs. |
| `DELETE` | `/api/projects/{project_id}/autopilot/sessions/{session_id}` | Delete a saved session. |
| `GET` | `/api/projects/{project_id}/autopilot/sessions/{session_id}/events` | Stream new session steps as Server-Sent Events. |
| `POST` | `/api/projects/{project_id}/autopilot/sessions/{session_id}/answers` | Submit answers when waiting for input. |
| `POST` | `/api/projects/{project_id}/autopilot/sessions/{session_id}/messages` | Continue a completed session with follow-up instructions. |
| `GET` | `/api/projects/{project_id}/autopilot/sessions/{session_id}/notebook` | Download the generated Jupyter notebook. |

Start a run:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/projects/customer_churn/autopilot/sessions" `
  -ContentType "application/json" `
  -Body '{"user_goal":"Predict customer churn and explain the strongest drivers."}'
```

Example response:

```json
{
  "project_id": "customer_churn",
  "session_id": "session_20260523_071500",
  "status": "running",
  "pending_step": null,
  "error": null,
  "worker_alive": true,
  "links": {
    "session": "/api/projects/customer_churn/autopilot/sessions/session_20260523_071500",
    "events": "/api/projects/customer_churn/autopilot/sessions/session_20260523_071500/events",
    "answers": "/api/projects/customer_churn/autopilot/sessions/session_20260523_071500/answers",
    "messages": "/api/projects/customer_churn/autopilot/sessions/session_20260523_071500/messages",
    "notebook": "/api/projects/customer_churn/autopilot/sessions/session_20260523_071500/notebook"
  }
}
```

Read the session:

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/projects/customer_churn/autopilot/sessions/session_20260523_071500"
```

Stream events:

```powershell
curl.exe -N "http://127.0.0.1:8000/api/projects/customer_churn/autopilot/sessions/session_20260523_071500/events"
```

If the event stream returns `waiting_for_input`, inspect `pending_step` for the
questions and submit answers:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/projects/customer_churn/autopilot/sessions/session_20260523_071500/answers" `
  -ContentType "application/json" `
  -Body '{"answers":["Use churn as the target and optimize recall."]}'
```

Send a follow-up after completion:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/projects/customer_churn/autopilot/sessions/session_20260523_071500/messages" `
  -ContentType "application/json" `
  -Body '{"message":"Try one more tuning round focused on reducing false negatives."}'
```

Download the generated notebook:

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8000/api/projects/customer_churn/autopilot/sessions/session_20260523_071500/notebook" `
  -OutFile "customer_churn_autopilot.ipynb"
```

## Event Stream Format

The events endpoint returns Server-Sent Events.

Step event:

```text
event: step
data: {"index":1,"kind":"thought","title":"AIML Scientist - Reasoning",...}
```

Status event:

```text
event: status
data: {"status":"complete","error":null}
```

Heartbeat event:

```text
event: heartbeat
data: {"status":"running"}
```

Each `step` payload is the same serialized `AutopilotStep` structure used by the
Streamlit UI session history.

## Persistence and Restart Behavior

Autopilot session files are saved under the project storage directory:

```text
{project_home}/{project_id}/autopilot/{session_id}/
```

Saved files include session metadata, streamed steps, LLM messages, notebook
entries, generated datasets, and training runs.

Important limitation: a paused `ask_user` step depends on the live Python
generator inside the API process. If the API server restarts while a session is
waiting for answers, the saved session can still be viewed, but the active pause
cannot be resumed. Start a new run, or continue from a completed/idle session.

## Notes for Clients

- Poll `GET /session` for simple integrations.
- Use `GET /events` for live progress updates.
- Store the returned `session_id`; it is the handle for answers, follow-ups, and downloads.
- Do not delete a session while it is running.
- The API is local-first and does not add authentication by itself. Put it behind
  your own auth layer before exposing it beyond your machine or private network.
