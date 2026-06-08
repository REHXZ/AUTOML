# aiml-discovery — Python Package

Packages the AUTOML autopilot engine as an installable Python library + CLI
so users can embed AutoML directly in their own pipelines without running the
Flask server or React UI.

---

## Install

```bash
pip install aiml-discovery
```

---

## CLI quickstart

```bash
# 1. Create a .env file with your LLM credentials
aiml-discovery init

# 2. Run the autopilot on a local dataset
aiml-discovery run --data ./data.csv --goal "Predict churn"

# 3. Run with a specific provider
aiml-discovery run --data ./data.csv --goal "Predict churn" --provider anthropic

# 4. Save the generated notebook + reports to an output folder
aiml-discovery run --data ./data.csv --goal "Predict churn" --output ./results/

# 5. Resume a previous session
aiml-discovery run --resume <session-id>
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data`, `-d` | required | Path to dataset (CSV, Excel, JSON, SQLite) |
| `--goal`, `-g` | required | Natural-language objective for the autopilot |
| `--provider` | auto | `openai` · `anthropic` · `azure` · `ollama` · `custom` |
| `--model` | provider default | Override the LLM model name |
| `--output`, `-o` | `./aiml_output/` | Directory for notebook + reports |
| `--project` | data filename | Project label (for session grouping) |
| `--resume` | — | Resume a previous run by session ID |
| `--quiet`, `-q` | off | Suppress step-by-step output |
| `--no-color` | off | Plain text output (good for CI logs) |

---

## Python API

### Simple one-shot run

```python
from aiml_discovery import Autopilot

pilot = Autopilot(
    data="./data.csv",
    goal="Predict churn",
)

for step in pilot.run():
    print(f"[{step.agent}] {step.thought or ''}")

# After completion
print("Best model:", pilot.best_model)
print("Training runs:", len(pilot.training_runs))
pilot.save_notebook("./results/notebook.ipynb")
```

### Bring your own credentials

```python
from aiml_discovery import Autopilot
from aiml_discovery.providers import ProviderConfig

pilot = Autopilot(
    data="./data.csv",
    goal="Predict churn",
    provider=ProviderConfig(
        provider="anthropic",
        api_key="sk-ant-...",
        model="claude-opus-4-8",
    ),
)
```

### Embedding in a pipeline (collect results without printing)

```python
from aiml_discovery import Autopilot

pilot = Autopilot(data="./sales.xlsx", goal="Forecast next quarter revenue")
steps = list(pilot.run())                    # blocking — waits for completion

summary = pilot.strategy_summary
runs = pilot.training_runs                   # list of dicts with model + metrics
pilot.save_notebook("./pipeline_output.ipynb")
```

### Interactive user-in-the-loop

```python
from aiml_discovery import Autopilot

pilot = Autopilot(data="./data.csv", goal="Detect fraud transactions")

for step in pilot.run():
    if step.kind == "ask_user":
        # The autopilot paused to ask a question
        answer = input(f"\n[Question] {step.thought}\nYour answer: ")
        for follow_up in pilot.answer(answer):
            print(f"[{follow_up.agent}] {follow_up.thought or ''}")
```

---

## Package layout

```
aiml_discovery/
    __init__.py          # Autopilot class, public re-exports
    cli.py               # `aiml-discovery` entry point (click)
    providers.py         # re-exports ProviderConfig from backend
backend/                 # existing backend (shipped inside the package)
    ...
pyproject.toml           # package metadata + entry_points
```

---

## Provider setup

The provider is auto-detected from environment variables in this order:

| Env var | Provider selected |
|---------|------------------|
| `ANTHROPIC_API_KEY` | anthropic |
| `OPENAI_API_BASE` + `OPENAI_API_KEY` | azure |
| `OPENAI_API_KEY` | openai |
| `OLLAMA_BASE_URL` | ollama (local, no key needed) |

Run `aiml-discovery init` to generate a `.env` template for any provider.

---

## Output files

After a run, `--output ./results/` will contain:

```
results/
    notebook.ipynb          # fully executable Jupyter notebook
    session.json            # step-by-step log (JSON)
    training_runs.json      # model metrics for every AutoML run
    datasets/               # any engineered datasets produced by the agents
```
