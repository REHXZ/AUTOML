# AI Autopilot

The AI Autopilot is a multi-agent AutoML system that takes your dataset and a plain-English goal, then autonomously runs the full ML discovery pipeline — from exploration to a trained, reviewed, and iterated model — with minimal input from you.

---

## Architecture Overview

```
UI (app.py)
    │
    ▼
AiAutopilot  (ai_autopilot.py)           ← thin façade; handles sessions
    │
    ▼
AimlScientist  (agents/scientist.py)     ← orchestrator LLM
    │
    ├──► EdaAgent                        ← data profiling & charts
    ├──► FeatureEngineeringAgent         ← dataset transformations
    ├──► ModelingAgent                   ← AutoML training runs
    ├──► ReviewAgent                     ← critiques results
    ├──► FineTuningAgent                 ← iterates on improvements
    └──► ResearcherAgent                 ← web search for domain gaps
```

All agents share a single `AgentContext` (notebook, datasets, training runs) and stream `AutopilotStep` events up to the UI in real time.

---

## How a Run Works

### 1. Start or Resume

`AiAutopilot.__init__` either starts a fresh session or reloads a prior one from disk. Resumed sessions restore the LLM conversation history, notebook, datasets, and training runs so the scientist can continue exactly where it left off.

### 2. The Scientist Orchestrates

`AimlScientist.run()` sends an initial prompt to the orchestrator LLM that includes:

- Project name
- Dataset index (ids, row/col counts, types)
- The user's stated goal

The Scientist then loops (up to 60 iterations), deciding at each step which specialist to call next.

### 3. Typical Flow

The Scientist is not locked to a fixed sequence — it uses its judgement — but the default happy path is:

```
ask_user?            ← only if the target column is genuinely ambiguous
    ↓
delegate_to_researcher  (optional — background domain research)
    ↓
delegate_to_eda      ← profiles every dataset, produces charts
    ↓
record_observation   ← Scientist's reading of the EDA
    ↓
delegate_to_feature_engineering  ← builds 1-2 cleaned/derived datasets
    ↓
delegate_to_modeling ← baseline AutoML run
    ↓
delegate_to_review   ← critiques the baseline
    ↓
delegate_to_fine_tuning  ← tries the top recommendations
    ↓
  (loop Review → Fine Tuning at least twice, until < 1 % metric gain)
    ↓
finalize_strategy    ← comprehensive markdown report, run ends
```

### 4. What Each Agent Does

| Agent | Role |
|---|---|
| **EDA Agent** | Profiles columns, detects types and missingness, produces Plotly charts. Has vision — it can "see" the charts it generates and comment on them. |
| **Feature Engineering Agent** | Applies transformations: drop/select columns, one-hot encoding, log transforms, bin numeric, interactions, polynomial features, groupby-aggregate, lag/lead/rolling windows for time series, dense panel construction, and raw Python execution. |
| **Modeling Agent** | Runs AutoML training via `train_automl`. Supports random splits (classification/regression) and chronological holdout (time-series forecasting). Produces leaderboard, feature-importance, predicted-vs-actual, and residual charts. |
| **Review Agent** | Reads training run metrics, leaderboard, and notebook. Identifies issues (leakage, underfitting, concept drift) and proposes concrete next experiments. |
| **Fine Tuning Agent** | Acts on Review's recommendations: retrains with revised features, hyperparameters, or different datasets and compares against the prior best. |
| **Researcher Agent** | Searches the web (via SearXNG) to fill domain knowledge gaps, look up ML technique benchmarks, or clarify unfamiliar data encodings. Findings are appended to the shared notebook. |

### 5. Asking the User

The Scientist calls `ask_user` sparingly — only when the answer materially changes the plan and cannot be inferred from the data. Examples:

- Which of two plausible target columns to optimise
- A business-side trade-off (false-positive cost vs. false-negative cost)
- Domain-specific definitions only the user can provide

Every question includes the Scientist's own recommendation and 1-2 alternatives so the user can accept the default with a single click.

### 6. Stopping Criteria

The Scientist keeps iterating until:

- At least 2 Review + Fine Tuning rounds have run, **and**
- The last round's best metric improved by < 1 % over the previous best (plateaued), **or**
- Review flags the data as at ceiling quality with no further levers available

### 7. Session Persistence

Every `AutopilotStep` is streamed to disk as it is produced (`steps.jsonl`). The full LLM conversation is also persisted (`messages.jsonl`). If the browser refreshes mid-run, passing the existing `session_id` to `AiAutopilot` rehydrates everything and `continue_with()` picks up seamlessly.

Session files live under:

```
{project_home}/{project_id}/autopilot/
    sessions.json           ← index of all sessions
    {session_id}/
        session.json        ← metadata + strategy summary
        steps.jsonl         ← streamed UI events
        messages.jsonl      ← LLM conversation history
        notebook.json       ← shared scratchpad entries
        new_datasets.json   ← datasets produced during the session
        training_runs.json  ← training run records
```

### 8. Final Output

When `finalize_strategy` is called the Scientist produces a comprehensive markdown report covering:

- The experiments run and their metrics
- What worked and what didn't
- The best model (quoted `run_id`)
- Recommended next steps

This report is displayed in the UI and persisted to `session.json`.

---

## Key Files

| File | Purpose |
|---|---|
| [aiml_discovery/ai_autopilot.py](../aiml_discovery/ai_autopilot.py) | Public façade — entry point for the UI |
| [aiml_discovery/agents/scientist.py](../aiml_discovery/agents/scientist.py) | Orchestrator LLM loop and tool dispatch |
| [aiml_discovery/agents/base.py](../aiml_discovery/agents/base.py) | `AgentContext`, `AutopilotStep`, `BaseAgent`, shared utilities |
| [aiml_discovery/agents/eda_agent.py](../aiml_discovery/agents/eda_agent.py) | EDA Agent |
| [aiml_discovery/agents/feature_engineering_agent.py](../aiml_discovery/agents/feature_engineering_agent.py) | Feature Engineering Agent |
| [aiml_discovery/agents/modeling_agent.py](../aiml_discovery/agents/modeling_agent.py) | Modeling Agent |
| [aiml_discovery/agents/review_agent.py](../aiml_discovery/agents/review_agent.py) | Review Agent |
| [aiml_discovery/agents/fine_tuning_agent.py](../aiml_discovery/agents/fine_tuning_agent.py) | Fine Tuning Agent |
| [aiml_discovery/agents/researcher_agent.py](../aiml_discovery/agents/researcher_agent.py) | Researcher Agent |
| [aiml_discovery/session_store.py](../aiml_discovery/session_store.py) | Session persistence (read/write/resume) |
| [aiml_discovery/training.py](../aiml_discovery/training.py) | `train_automl` — the AutoML training backend |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_BASE` | — | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-5.4` | Deployment name used by all agents |
| `OPENAI_API_KEY` | — | API key (can also come from `.env`) |
