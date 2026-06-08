# AI Autopilot

The AI Autopilot is a multi-agent AutoML system that takes your dataset and a plain-English goal, then autonomously runs the full ML discovery pipeline — from exploration to a trained, reviewed, and iterated model — with minimal input from you.

---

## Architecture Overview

```
AiAutopilot  (backend/logic/autopilot.py)          ← thin façade; handles sessions
    │  registers HookManager on AgentContext
    │
    ▼
AimlScientist  (backend/logic/agents/scientist.py) ← orchestrator LLM (up to 60 iters)
    │  fires BEFORE_DELEGATE / AFTER_DELEGATE
    │
    ├──► EdaAgent                    ← data profiling & charts
    ├──► FeatureEngineeringAgent     ← 50+ dataset transformations
    ├──► ModelingAgent               ← AutoML training (25+ models)
    ├──► ModelTesterAgent            ← held-out test-set evaluation
    ├──► ReviewAgent                 ← critiques runs, flags leakage
    ├──► FineTuningAgent             ← iterates on improvements
    ├──► ResearcherAgent             ← web search for domain gaps
    └──► DriftAgent                  ← distributional shift detection

Each sub-agent call passes through the Hook Lifecycle:
    BEFORE_DELEGATE → [agent runs] → AFTER_DELEGATE (steering / retry)
    BEFORE_LLM      → [API call]   → AFTER_LLM (token accounting)
    BEFORE_TOOL     → [dispatch]   → AFTER_TOOL (guardrails / modify)
```

All agents share a single `AgentContext` (notebook, datasets, training runs, hook manager) and stream `AutopilotStep` events up to the UI in real time through `AiAutopilot._tee()`.

---

## Hook Lifecycle System

Every LLM call and tool dispatch in every agent passes through a shared `HookManager` stored on `AgentContext.hooks`. This enables cross-cutting behaviour — observability, flow control, guardrails — to be registered once as independent policies rather than duplicated inline.

### Lifecycle events

| Event | Fires when | Can control |
|---|---|---|
| `BEFORE_LLM` | Before each `chat.completions.create()` | Abort the loop |
| `AFTER_LLM` | After a successful API response | Observe / accumulate |
| `BEFORE_TOOL` | Before `dispatch(name, args)` | Modify args, Skip, Abort |
| `AFTER_TOOL` | After dispatch returns | Modify result |
| `TOOL_ERROR` | When dispatch raises | Skip (recover with canned result) |
| `BEFORE_DELEGATE` | Before Scientist spawns a sub-agent | Modify instructions, Skip, Abort |
| `AFTER_DELEGATE` | After sub-agent `.run()` returns | Retry with new instructions |
| `STEP_EMITTED` | Every `AutopilotStep` through `_tee` | Observe only |
| `RUN_START` / `RUN_END` | Agent `.run()` entry / exit | Observe only |
| `ASK_USER` | Scientist emits an `ask` pause | Observe only |

### Hook outcomes

A hook returns a `HookOutcome` with one of five decisions (merged in precedence order):

```
CONTINUE (0) < MODIFY (1) < SKIP (2) < RETRY (3) < ABORT (4)
```

- **CONTINUE** — do nothing special.
- **MODIFY** — rewrite tool `args`, delegation `instructions`, or `result`.
- **SKIP** — short-circuit: use the hook's canned `result` instead of calling the real action.
- **RETRY** — re-run the sub-agent (delegation only) with optionally new `instructions`.
- **ABORT** — break the enclosing LLM loop immediately.

### Built-in policies (`backend/logic/agents/hook_policies.py`)

| Hook | Priority | Event(s) | Replaces |
|---|---|---|---|
| `StopHook` | 1 | `BEFORE_LLM` | Inline `should_stop` checks (duplicated in 2 loops) |
| `GuardrailHook` | 5 | `BEFORE_TOOL` | — (new capability) |
| `TokenAccountingHook` | 10 | `AFTER_LLM` | Inline `response.usage` blocks (duplicated in 2 loops) |
| `LoggingHook` | 20 | all major events | Scattered `log.info/debug` calls |
| `ModelTesterGateHook` | 30 | `BEFORE_DELEGATE` | Hardcoded "run Tester before Review" block |
| `SteeringHook` | 50 | `AFTER_DELEGATE` | `_steer_check` / `_delegate_with_steering` |

All six are registered by `default_hook_manager()`, which `AiAutopilot.__init__` calls automatically.

### Writing a custom hook

```python
from backend.logic.agents.hooks import Hook, HookContext, HookEvent, HookOutcome

class AuditHook(Hook):
    name = "audit"
    priority = 90          # run late (after built-in hooks)
    events = frozenset({HookEvent.AFTER_TOOL, HookEvent.AFTER_DELEGATE})

    def handle(self, hc: HookContext):
        # Hooks are generators — yield AutopilotStep objects if needed.
        yield from ()
        print(f"[audit] agent={hc.agent_name} event={hc.event.value}")
        return HookOutcome.cont()
```

Register before starting a run:

```python
from backend.logic.agents import default_hook_manager
from backend.logic.autopilot import AiAutopilot

pilot = AiAutopilot(api_key=..., project_id=..., store=...)
pilot._ctx.hooks.register(AuditHook())
```

---

## How a Run Works

### 1. Start or Resume

`AiAutopilot.__init__` either starts a fresh session or reloads a prior one from disk. Resumed sessions restore the LLM conversation history, notebook, datasets, and training runs so the scientist can continue exactly where it left off.

On every start (fresh or resumed) a `HookManager` is built with the default policies and attached to `AgentContext.hooks`. The OpenAI client and deployment name are also stored on `AgentContext` so hook policies can spawn sub-agents (e.g. `ModelTesterGateHook`).

### 2. The Scientist Orchestrates

`AimlScientist.run()` sends an initial prompt to the orchestrator LLM that includes:

- Project name and dataset index (ids, row/col counts, types)
- The user's stated goal

The Scientist loops (up to 60 iterations), deciding at each step which specialist to call next. Every tool call fires `BEFORE_TOOL` / `AFTER_TOOL`; every sub-agent delegation fires `BEFORE_DELEGATE` / `AFTER_DELEGATE`.

### 3. Typical Flow

The Scientist is not locked to a fixed sequence — it uses its judgement — but the default happy path is:

```
ask_user?              ← only if the target column is genuinely ambiguous
    ↓
delegate_to_researcher  (optional — background domain research)
    ↓
delegate_to_eda        ← profiles every dataset, produces charts
    ↓  (SteeringHook may retry with refined instructions)
record_observation     ← Scientist's reading of the EDA
    ↓
delegate_to_feature_engineering  ← builds 1-2 cleaned/derived datasets
    ↓
delegate_to_modeling   ← baseline AutoML run
    ↓
delegate_to_model_tester  ← held-out test evaluation
    ↓  (ModelTesterGateHook also fires here when review is next)
delegate_to_review     ← critiques the baseline
    ↓
delegate_to_fine_tuning  ← tries top recommendations
    ↓
  (loop Review → Fine Tuning at least twice, until < 1 % metric gain)
    ↓
finalize_strategy      ← comprehensive markdown report, run ends
```

### 4. What Each Agent Does

| Agent | Role |
|---|---|
| **EDA Agent** | Profiles columns, detects types and missingness, produces Plotly charts. Has vision — it can "see" the charts it generates and comment on them. |
| **Feature Engineering Agent** | Applies 50+ transformations: encoding, imputation, scaling, binning, interactions, polynomial features, groupby-aggregate, lag/lead/rolling windows, dense panel construction. |
| **Modeling Agent** | Runs AutoML training via `train_automl`. Supports random splits (classification/regression) and chronological holdout (time-series). Produces leaderboard, feature-importance, and diagnostic charts. |
| **Model Tester Agent** | Loads saved pipelines and evaluates them against held-out test CSVs. Always runs before Review so the Review Agent sees real out-of-sample metrics. |
| **Review Agent** | Reads training run metrics, leaderboard, and notebook. Identifies issues (leakage, underfitting, concept drift) and proposes concrete next experiments. |
| **Fine Tuning Agent** | Acts on Review's recommendations: retrains with revised features, hyperparameters, or different datasets and compares against the prior best. |
| **Researcher Agent** | Searches the web to fill domain knowledge gaps, look up ML technique benchmarks, or clarify unfamiliar data encodings. Findings are appended to the shared notebook. |
| **Drift Agent** | Detects distributional shifts (PSI / KS test) between a reference and production dataset. |

### 5. Asking the User

The Scientist calls `ask_user` sparingly — only when the answer materially changes the plan and cannot be inferred from the data. Every question includes the Scientist's own recommendation and 1–2 alternatives so the user can accept the default with a single click.

### 6. Stopping Criteria

The Scientist keeps iterating until:

- At least 2 Review + Fine Tuning rounds have run, **and**
- The last round's best metric improved by < 1 % over the previous best, **or**
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
| `backend/logic/autopilot.py` | Public façade — session lifecycle, hook wiring, `_tee` streaming |
| `backend/logic/agents/scientist.py` | Orchestrator LLM loop, `_delegate()`, tool dispatch |
| `backend/logic/agents/base.py` | `AgentContext`, `AutopilotStep`, `BaseAgent`, `_fire`, `_invoke_llm` |
| `backend/logic/agents/hooks.py` | `HookEvent`, `HookOutcome`, `Hook`, `HookManager` — hook framework |
| `backend/logic/agents/hook_policies.py` | Built-in policies + `default_hook_manager()` |
| `backend/logic/agents/eda_agent.py` | EDA Agent |
| `backend/logic/agents/feature_engineering_agent.py` | Feature Engineering Agent |
| `backend/logic/agents/modeling_agent.py` | Modeling Agent |
| `backend/logic/agents/model_tester.py` | Model Tester Agent |
| `backend/logic/agents/review_agent.py` | Review Agent |
| `backend/logic/agents/fine_tuning_agent.py` | Fine Tuning Agent |
| `backend/logic/agents/researcher_agent.py` | Researcher Agent |
| `backend/logic/agents/drift_agent.py` | Drift Agent |
| `backend/services/session_store.py` | Session persistence (read / write / resume) |
| `backend/logic/training.py` | `train_automl` — AutoML training backend |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_BASE` | — | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-5.4` | Deployment name used by all agents |
| `OPENAI_API_KEY` | — | API key (can also come from `.env`) |
| `AIML_DISCOVERY_HOME` | `~/.aiml_discovery` | Root directory for project artifacts |
