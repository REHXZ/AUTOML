# AIML Autopilot — Agent Playbook

> **How the system thinks, how to steer it, and how to fix it when something goes wrong.**

---

## 1. How the Autopilot Solves a Problem

The autopilot follows a modified CRISP-DM lifecycle, managed by the **AIML Scientist** (orchestrator). Every run moves through these phases:

```
Business Understanding → Data Understanding → Data Preparation
        → Modeling → Evaluation → Iteration → Finalize
```

### The Orchestrator's Planning Step
Before any delegation, the Scientist writes a **Discovery Plan** into the shared notebook. This plan defines:
- Task type (Classification / Regression / Time-Series Forecasting)
- Primary metric and success threshold  
- Data assessment (target column, time column, key risks)
- Feature engineering strategy
- Modeling strategy and stop criteria
- 2–3 testable hypotheses

Every sub-agent receives this context when it reads the notebook.

---

## 2. Agent Roles and Expert Focus

| Agent | Expert Focus | When It's Called |
|-------|-------------|-----------------|
| **EDA Agent** | Statistical profiling, leakage detection, distribution analysis | Phase 2: Data Understanding |
| **FE Agent** | 50+ transformations, time-series features, resampling | Phase 3: Data Preparation |
| **Modeling Agent** | AutoML (25+ models), ARIMA, SHAP, Optuna | Phase 4: Modeling |
| **Review Agent** | Leakage audit, overfitting check, improvement ranking | Phase 5: Evaluation |
| **Fine Tuning Agent** | Strategic experimentation, metric improvement | Phase 4/6: Iteration |
| **Researcher Agent** | Web search for domain knowledge and benchmarks | Any phase |
| **Drift Agent** | PSI/KS-test distributional comparison | Monitoring phase |

---

## 3. Common Issues and How to Resolve Them

### Issue: Model metrics look too good (R² > 0.98, AUC > 0.99)
**Diagnosis:** Data leakage — a feature is directly derived from or correlated with the target.

**Resolution steps:**
1. Ask the Scientist to run the Review Agent and check feature importances.
2. Look for: (a) a single feature dominating importance, (b) features with correlation > 0.95 to target.
3. Drop the suspect feature via `drop_columns` in the FE Agent.
4. Retrain. Expect the metric to DROP — this is correct. The honest lower metric is more useful.

**Prevention:** EDA Agent always runs `target_correlation` and `mutual_information` before FE.

---

### Issue: All models match the baseline (no signal detected)
**Diagnosis:** Wrong target column, wrong aggregation grain, or target has no predictors.

**Resolution steps:**
1. Check: is the target column in the modelling dataset? (Use `inspect_dataset`.)
2. If target is missing: rewind to FE, add `groupby_aggregate` with the correct grouping.
3. Check aggregation grain: predicting per-row when the real target is monthly? → Re-aggregate.
4. Check: does the target have variance? A constant target cannot be modelled.
5. If all features are noise: ask the Researcher to look up feature engineering ideas for this domain.

---

### Issue: Time-series predictions are flat (predicting the mean)
**Diagnosis:** Missing lag/rolling features, or wrong training mode (random split on temporal data).

**Resolution steps:**
1. Verify the Modeling Agent used `time_column` for chronological holdout.
2. Verify the dataset has lag/rolling/lead columns from FE. If not: rewind to FE.
3. FE pipeline needed: `groupby_aggregate` → `dense_panel` → `create_lag_features` → `create_rolling_features` → `create_lead_target`.
4. After FE: retrain with `time_column` set so the holdout is the last N% of time periods.

---

### Issue: Classification F1 is low despite high accuracy
**Diagnosis:** Class imbalance — the model is predicting the majority class.

**Resolution steps:**
1. Run EDA Agent `class_balance` analysis — confirm the imbalance ratio.
2. If imbalance > 3:1: rewind to FE, apply `smote_tomek` (best combined approach).
3. In Modeling Agent: set `class_weight="balanced"` in `train_model`.
4. Use F1_weighted and AUC-ROC as metrics, not accuracy.

---

### Issue: Feature Engineering operations returning no output / empty datasets
**Diagnosis:** Parameters passed at the wrong level (top-level instead of inside `params`).

**Resolution:**
All per-operation arguments MUST be nested inside the `params` object:
```json
{
  "source_dataset_id": "abc123",
  "new_name": "monthly_panel",
  "operation": "groupby_aggregate",
  "params": {
    "group_by": ["month", "product_id"],
    "aggregations": {"qty": "sum", "orders": "nunique"}
  }
}
```
❌ Wrong: `"group_by": ["month"]` at the top level.
✅ Right: `"params": {"group_by": ["month"]}`.

---

### Issue: Drift detected in production — what to do?
**Step 1:** Run Drift Detection Agent with `run_drift_report`.  
**Step 2:** Classify drift type (covariate / label / concept / pipeline).  
**Step 3:**
- **Covariate drift only** (features shifted, target stable): Monitor; consider periodic retraining.
- **Label drift** (target distribution changed): Retrain on recent data immediately.
- **Concept drift** (P(Y|X) changed): Full pipeline rerun with recent data as training set.
- **Pipeline drift** (null rates spiked, encoding changed): Investigate upstream data source.

**PSI > 0.20 on any feature** = retrain recommended.  
**PSI > 0.35** = model performance likely severely degraded; retrain urgently.

---

### Issue: ARIMA/SARIMA not fitting well
**Diagnosis:** Non-stationary series, missing seasonality detection, or wrong order.

**Resolution steps:**
1. EDA Agent: run `stationarity_test` (ADF + KPSS) first.
2. If non-stationary: FE adds differencing or `target_log_transform`.
3. EDA Agent: run `acf_pacf` to determine lag order p and q.
4. Modeling Agent: call `train_arima` — it auto-selects order via AIC.
5. Use `arima_forecast` to produce n-step-ahead forecasts with confidence intervals.

---

### Issue: Optuna HPO takes too long
**Resolution:** Reduce `n_trials` (default 30). For a quick search use 15 trials.  
Optuna is Bayesian — 15 good trials > 100 random (RandomizedSearchCV).  
For very large datasets, use `cross_val_score` with 3 folds to speed up evaluation.

---

## 4. Metric Reference Guide

| Task | Primary Metric | Secondary | Notes |
|------|---------------|-----------|-------|
| Binary classification (balanced) | AUC-ROC | F1 | AUC > 0.85 is strong |
| Binary classification (imbalanced) | F1_weighted | Recall (minority) | Accuracy is misleading |
| Multi-class classification | F1_weighted | F1 per class | Check per-class recall |
| Regression | R² | RMSE, MAE | R² > 0.75 is strong for tabular |
| Time-series forecast | RMSE | MAE, MAPE | Always use chronological holdout |
| Demand forecast | MAPE | RMSE | < 10% MAPE is excellent |

**Suspicious metric thresholds** (probable leakage):
- Classification: AUC > 0.99, F1 > 0.97
- Regression: R² > 0.98
- Any task: train metric >> test metric by > 15%

---

## 5. Feature Engineering Decision Guide

| Situation | Recommended Operation |
|-----------|----------------------|
| Target is right-skewed (skewness > 1.5) | `target_log_transform` in-place |
| Feature is right-skewed | `log_transform` (adds new column) |
| Category with > 50 unique values | `target_encode` or `hash_encode` |
| Category with ≤ 20 unique values | `one_hot_encode` |
| Month / hour / day-of-week | `cyclical_encode` (sin/cos) |
| Missing values > 50 % | `drop_high_missing` |
| Missing values 5-50 % | `knn_impute` + `add_missing_indicators` |
| Outlier rate > 10 % | `winsorize` (clip to 1st-99th percentile) |
| Multicollinearity (VIF > 10) | `drop_correlated` threshold=0.9 |
| Forecasting on transactional data | `groupby_aggregate` → `dense_panel` → `create_lag_features` → `create_lead_target` |
| Class imbalance > 5:1 | `smote_tomek` (combined over- and under-sampling) |
| Text columns | `text_clean` → `tfidf_vectorize` or `lsa_text` |
| Too many features (> 50) | `select_from_model` or `rfe_select` |

---

## 6. Reading the Exported Notebook

The exported Jupyter notebook is structured by CRISP-DM phase. Each cell is tagged with the phase it belongs to. The structure is:

1. **Business Understanding** — problem statement, discovery plan
2. **Data Understanding** — EDA charts, statistical findings, leakage checks
3. **Data Preparation** — FE operations applied, dataset lineage
4. **Modeling** — training runs, leaderboard, diagnostic charts
5. **Evaluation** — Review Agent critique, improvement rankings
6. **Iteration** — fine-tuning rounds, metric progression
7. **Final Strategy** — best model summary, recommendations

All Plotly charts are runnable in the notebook with the exact same code used during the autopilot run.

---

## 7. How to Steer the Autopilot

The Scientist asks questions only when genuinely ambiguous. For maximum control, include in your initial prompt:

- **Target column name** (or description if it needs to be derived)
- **Task type** (classification / regression / forecasting)
- **Primary metric** (e.g., "optimise for F1 on the positive class")
- **Business constraints** (e.g., "false positives are 5× more costly than false negatives")
- **Time column** (for forecasting)
- **Forecast horizon** (e.g., "predict 1, 3, and 6 months ahead")

The more context you provide upfront, the fewer questions the Scientist asks and the more focused the analysis becomes.

---

## 8. Extending the System with Hooks

The autopilot exposes a **hook lifecycle** at every major control point (LLM calls, tool dispatches, sub-agent delegations). Hooks let you add cross-cutting behaviour — auditing, rate-limiting, domain guardrails, custom steering logic — without modifying any agent code.

### Quick start

```python
from backend.logic.agents.hooks import Hook, HookContext, HookEvent, HookOutcome
from backend.logic.autopilot import AiAutopilot

class CostGuardHook(Hook):
    """Abort the run if cumulative prompt tokens exceed a budget."""
    name = "cost_guard"
    priority = 2                # run immediately after StopHook
    events = frozenset({HookEvent.BEFORE_LLM})

    def __init__(self, max_tokens: int = 500_000):
        self._max = max_tokens

    def handle(self, hc: HookContext):
        yield from ()
        total = sum(
            u.get("prompt_tokens", 0)
            for u in hc.ctx.agent_token_usage.values()
        )
        if total >= self._max:
            return HookOutcome.abort(reason=f"token budget {self._max} exceeded (used {total})")
        return HookOutcome.cont()

# Register before the first step is produced:
pilot = AiAutopilot(api_key=..., project_id=..., store=...)
pilot._ctx.hooks.register(CostGuardHook(max_tokens=200_000))
```

### Hook event reference

See `backend/logic/agents/hooks.py` for the full `HookEvent` enum and `HookContext` dataclass fields. See `backend/logic/agents/hook_policies.py` for the six built-in hooks as worked examples.

### Outcome decision precedence

When multiple hooks fire on the same event their outcomes are merged:

```
ABORT (4) > RETRY (3) > SKIP (2) > MODIFY (1) > CONTINUE (0)
```

Two `MODIFY` outcomes on the same event accumulate their fields (later hook's non-None fields override earlier ones).

### Built-in hooks you can subclass

| Class | File | Common reason to subclass |
|---|---|---|
| `GuardrailHook` | `hook_policies.py` | Add domain-specific `BLOCKED_TOOLS` or column validation |
| `SteeringHook` | `hook_policies.py` | Change steering LLM prompt or satisfaction threshold |
| `LoggingHook` | `hook_policies.py` | Route logs to a remote sink (Datadog, CloudWatch) |

---

*Last updated: 2026-06-07*
