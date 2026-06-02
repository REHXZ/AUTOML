# Agent Tool Catalog

> **Living document.** This is the master list of every tool available to the
> AIML Autopilot agents. The goal is that the agents can train a model on **any**
> dataset by reaching for whatever technique a task requires — checking for
> seasonality, fixing class imbalance with SMOTE, selecting features, reducing
> dimensionality, and so on.
>
> All items marked ✅ are implemented, tested, and callable. Items marked 🔜
> are planned for a future iteration.

## Status legend

| Marker | Meaning |
| --- | --- |
| ✅ | Implemented, tested, and callable by the agent |
| 🔜 | Planned — not yet implemented |

## How tools reach the agents

Each agent exposes a small number of **LLM-facing tools** (OpenAI function-calling
schemas in each agent's `_tools()`), and most of the breadth lives behind a single
dispatch tool with an `operation` / `chart_type` / `analysis_type` selector:

- **Feature Engineering Agent** — one tool, `create_derived_dataset(operation, params, …)`.
  Every transformation below is an `operation`. The result is registered as a new
  dataset so the pipeline stays reproducible and auditable.
- **EDA Agent** — `profile_dataset`, `create_chart(chart_type, params)` (returns an
  image the agent can *see*), and `run_analysis(analysis_type, params)` (returns
  structured statistics, optionally with a chart).
- **Modeling Agent** — `train_model(...)` (AutoML across the full model catalogue),
  `cross_validate_model`, `tune_hyperparameters`, plus diagnostics tools.
- **Review / Fine-Tuning / Researcher / Scientist** — orchestration tools
  (delegation, web search, notebook notes). Their power comes from the workers above.

Optional heavy dependencies (`statsmodels`, `imbalanced-learn`, `category-encoders`,
`xgboost`, `lightgbm`, `catboost`) are imported lazily. If one is missing, the tool
returns a clear "install X to use this" message instead of crashing the run.

---

## 1. Feature Engineering Agent — `operation`s

### 1.1 Cleaning & row operations

| Operation | Purpose | Status |
| --- | --- | --- |
| `drop_high_missing` | Drop columns above a missing-fraction threshold | ✅ |
| `drop_duplicates` | Remove duplicate rows | ✅ |
| `select_columns` | Keep only the listed columns | ✅ |
| `drop_columns` | Remove the listed columns | ✅ |
| `filter_outliers` | IQR (1.5×) outlier row removal on numeric cols | ✅ |
| `drop_constant` | Drop zero-variance / single-value columns | ✅ |
| `drop_correlated` | Drop one of each highly-correlated numeric pair | ✅ |

### 1.2 Imputation (missing values)

| Operation | Purpose | Status |
| --- | --- | --- |
| `impute_missing` | Median (numeric) / mode (categorical) fill | ✅ |
| `constant_impute` | Fill chosen columns with a constant | ✅ |
| `knn_impute` | `KNNImputer` — neighbour-based numeric imputation | ✅ |
| `iterative_impute` | `IterativeImputer` (MICE-style) | ✅ |
| `add_missing_indicators` | Add `{col}_was_missing` boolean flags | ✅ |

### 1.3 Scaling & numeric transforms

| Operation | Purpose | Status |
| --- | --- | --- |
| `log_transform` | `log1p` of chosen columns (adds new cols) | ✅ |
| `target_log_transform` | `log1p` of a skewed regression target (in place) | ✅ |
| `standard_scale` | `StandardScaler` (z-score) | ✅ |
| `minmax_scale` | `MinMaxScaler` to [0, 1] | ✅ |
| `robust_scale` | `RobustScaler` (median/IQR, outlier-robust) | ✅ |
| `max_abs_scale` | `MaxAbsScaler` (sparse-friendly) | ✅ |
| `power_transform` | `PowerTransformer` (Yeo-Johnson / Box-Cox) | ✅ |
| `quantile_transform` | `QuantileTransformer` (uniform / normal output) | ✅ |
| `bin_numeric` | Quantile-bin a numeric column | ✅ |
| `clip_values` | Clip a column to a [min, max] range | ✅ |
| `winsorize` | Clip to lower/upper percentiles | ✅ |

### 1.4 Encoding (categoricals & dates)

| Operation | Purpose | Status |
| --- | --- | --- |
| `one_hot_encode` | One-hot encode low-cardinality categoricals | ✅ |
| `ordinal_encode` | `OrdinalEncoder` — integer codes | ✅ |
| `label_encode` | Integer-encode a single column | ✅ |
| `frequency_encode` | Replace categories with their frequency/count | ✅ |
| `target_encode` | Mean-of-target encoding (smoothed, uses category_encoders) | ✅ |
| `hash_encode` | Feature hashing for very high cardinality categoricals (IDs, tokens) | ✅ |
| `encode_dates` | Expand datetime → year/month/day/dow/quarter | ✅ |
| `datetime_parse` | Coerce a column to proper datetime dtype | ✅ |
| `cyclical_encode` | sin/cos of cyclic fields (month 1-12, dow 0-6, hour 0-23) | ✅ |
| `fourier_features` | Fourier (sin/cos harmonics) for seasonal/periodic columns | ✅ |
| `interaction_features` | Multiplicative pairs `a*b` | ✅ |
| `polynomial_features` | Squared / cubed terms | ✅ |
| `rename_columns` | Rename columns | ✅ |

### 1.5 Aggregation

| Operation | Purpose | Status |
| --- | --- | --- |
| `groupby_aggregate` | Group-by rollups (sum/mean/nunique/count/min/max/first/last) | ✅ |

### 1.6 Time-series features (forecasting)

| Operation | Purpose | Status |
| --- | --- | --- |
| `dense_panel` | Fill missing (group × time) combinations | ✅ |
| `create_lag_features` | Backward lags within group | ✅ |
| `create_rolling_features` | Leakage-safe rolling-window aggregates | ✅ |
| `create_lead_target` | Forward-shifted forecasting targets (t+1, t+3, …) | ✅ |

### 1.7 Feature selection & dimensionality reduction

| Operation | Purpose | Status |
| --- | --- | --- |
| `select_k_best` | `SelectKBest` (F-test or mutual information) | ✅ |
| `select_from_model` | `SelectFromModel` (RandomForest importance threshold) | ✅ |
| `rfe_select` | Recursive Feature Elimination | ✅ |
| `pca` | `PCA` dimensionality reduction | ✅ |

### 1.8 Outlier handling

| Operation | Purpose | Status |
| --- | --- | --- |
| `winsorize` | Clip to percentile range | ✅ |
| `clip_values` | Clip a column to explicit [min, max] | ✅ |
| `zscore_outlier_removal` | Drop rows with \|z\| above a threshold | ✅ |
| `isolation_forest_outliers` | `IsolationForest` anomaly removal | ✅ |
| `filter_outliers` | IQR row removal | ✅ |

### 1.9 Class-imbalance resampling (`imbalanced-learn`)

> Always set `params.target_column`. Impute missing values first.

| Operation | Purpose | Status |
| --- | --- | --- |
| `smote` | SMOTE (auto-SMOTENC when categoricals present) | ✅ |
| `borderline_smote` | Borderline-SMOTE (focus on decision border) | ✅ |
| `adasyn` | ADASYN adaptive oversampling | ✅ |
| `random_oversample` | Random minority oversampling | ✅ |
| `random_undersample` | Random majority undersampling | ✅ |
| `smote_tomek` | SMOTE + Tomek-link cleaning (combined) | ✅ |
| `smote_enn` | SMOTE + Edited-Nearest-Neighbours (combined) | ✅ |

### 1.10 Escape hatch

| Operation | Purpose | Status |
| --- | --- | --- |
| `execute_python` | Run arbitrary pandas/numpy on the dataframe | ✅ |

---

## 2. EDA Agent

### 2.1 Charts — `create_chart(chart_type, …)` (image returned to the agent)

| Chart type | Purpose | Status |
| --- | --- | --- |
| `histogram` | Single-column distribution | ✅ |
| `bar` | Top-N value counts | ✅ |
| `scatter` | x vs y (optional colour) | ✅ |
| `correlation_heatmap` | Numeric correlation matrix | ✅ |
| `box` | Box plot (optional group-by) | ✅ |
| `violin` | Violin plot (optional group-by) | ✅ |
| `pairplot` | Scatter matrix of up to 4 numerics | ✅ |
| `missing_heatmap` | Missing-value pattern | ✅ |
| `line` | Value(s) over an ordered/time axis | ✅ |
| `qq_plot` | Normal Q–Q plot (distribution normality check) | ✅ |

### 2.2 Analyses — `run_analysis(analysis_type, …)` (structured stats + optional chart)

| Analysis type | Purpose | Status |
| --- | --- | --- |
| `class_balance` | Target class counts + imbalance ratio | ✅ |
| `target_correlation` | Feature↔target Pearson correlation | ✅ |
| `mutual_information` | Non-linear feature-target MI scores | ✅ |
| `normality_test` | Shapiro / D'Agostino + skew & kurtosis | ✅ |
| `vif` | Variance Inflation Factor (multicollinearity) | ✅ |
| `outlier_summary` | IQR / z-score outlier counts per column | ✅ |
| `seasonal_decompose` | Trend / seasonal / residual decomposition (statsmodels) | ✅ |
| `stationarity_test` | ADF + KPSS stationarity tests | ✅ |
| `acf_pacf` | Autocorrelation & partial-autocorrelation with chart | ✅ |

---

## 3. Modeling Agent

### 3.1 Model catalogue (AutoML `train_model`)

All of the following are tried automatically (subset selectable via `include_models`);
custom estimators can be added via `custom_models`.

**Classification:** Baseline (Majority), Logistic Regression, SGD Classifier,
Linear SVC, Gaussian/Bernoulli/Multinomial NB, LDA, QDA, Decision Tree, Extra Trees,
Random Forest, AdaBoost, Gradient Boosting, Hist Gradient Boosting, KNN, SVC (RBF),
MLP, Bagging, XGBoost*, LightGBM*, CatBoost*. — ✅

**Regression:** Baseline (Mean), Linear Regression, Ridge, Lasso, ElasticNet,
Bayesian Ridge, Huber, SGD Regressor, Decision Tree, Extra Trees, Random Forest,
AdaBoost, Gradient Boosting, Hist Gradient Boosting, KNN, Linear SVR, SVR (RBF),
MLP, Bagging, XGBoost*, LightGBM*, CatBoost*. — ✅

`*` third-party; auto-included when the library is installed.

### 3.2 Training options & tools

| Capability | Purpose | Status |
| --- | --- | --- |
| Chronological split (`time_column`) | Honest forecasting backtest | ✅ |
| `include_models` / `custom_models` | Restrict / extend the catalogue | ✅ |
| Diagnostic charts | predicted-vs-actual, residuals, confusion, importance, leaderboard | ✅ |
| `compare_runs` | Side-by-side metric comparison bar chart | ✅ |
| `class_weight="balanced"` | Cost-sensitive training for imbalanced classification (in `train_model` and `cross_validate_model`) | ✅ |
| `cross_validate_model` | k-fold / TimeSeriesSplit CV with mean ± std scores | ✅ |
| `tune_hyperparameters` | RandomizedSearchCV over default or custom param grid | ✅ |
| `build_ensemble` | VotingClassifier / VotingRegressor or Stacking over selected models | ✅ |

---

## 4. Shared / orchestration tools

| Agent | Tools | Status |
| --- | --- | --- |
| Scientist | `set_phase`, `ask_user`, `delegate_to_*`, `record_observation`, `finalize_strategy` | ✅ |
| Review | `record_finding`, `spawn_researcher`, `done` (now cites FE + Modeling improvements) | ✅ |
| Fine Tuning | `spawn_feature_engineering`, `spawn_modeling`, `record_finding`, `done` | ✅ |
| Researcher | `search_web`, `fetch_page`, `record_finding`, `done` | ✅ |

---

## Dependencies

| Package | Purpose | Required? |
| --- | --- | --- |
| `scikit-learn >= 1.4` | Core ML (all standard models, scalers, selectors) | Required |
| `scipy >= 1.11` | Normality tests, Q-Q plot, statistical helpers | Required |
| `statsmodels >= 0.14` | Seasonal decomposition, ADF/KPSS, ACF/PACF, VIF | Required |
| `imbalanced-learn >= 0.12` | SMOTE family and other resamplers | Required |
| `category-encoders >= 2.6` | Smoothed target encoding | Required |
| `pandas >= 2.2` | DataFrames | Required |
| `plotly >= 5.22` | Interactive charts | Required |
| `xgboost >= 2.0` | XGBoost models | Optional (auto-detected) |
| `lightgbm >= 4.0` | LightGBM models | Optional (auto-detected) |
| `catboost >= 1.2` | CatBoost models | Optional (auto-detected) |

---

## Changelog

### 2026-06-02 — Tool expansion (complete)
- Added `statsmodels`, `imbalanced-learn`, `category-encoders`, `scipy` as required dependencies.
- **Feature Engineering** — 35 new operations:
  - Scaling: `standard_scale`, `minmax_scale`, `robust_scale`, `max_abs_scale`, `power_transform`, `quantile_transform`, `clip_values`, `winsorize`
  - Imputation: `constant_impute`, `knn_impute`, `iterative_impute` (MICE), `add_missing_indicators`
  - Encoding: `ordinal_encode`, `label_encode`, `frequency_encode`, `target_encode`, `hash_encode`, `cyclical_encode`, `fourier_features`, `datetime_parse`
  - Cleaning: `drop_constant`, `drop_correlated`
  - Selection/Reduction: `select_k_best`, `select_from_model`, `rfe_select`, `pca`
  - Outliers: `zscore_outlier_removal`, `isolation_forest_outliers`
  - Class imbalance: `smote`, `borderline_smote`, `adasyn`, `random_oversample`, `random_undersample`, `smote_tomek`, `smote_enn`
- **EDA** — new `run_analysis` tool with 9 analyses: `seasonal_decompose`, `stationarity_test`, `acf_pacf`, `class_balance`, `target_correlation`, `mutual_information`, `normality_test`, `vif`, `outlier_summary`. New charts: `line`, `qq_plot`.
- **Modeling** — `class_weight="balanced"` in both `train_model` and `cross_validate_model`; `cross_validate_model` (k-fold / TimeSeriesSplit); `tune_hyperparameters` (RandomizedSearchCV with built-in grids for 13 model families); `build_ensemble` (Voting or Stacking over any combination of trained models).
- **Review Agent** — updated to cite all new FE, Modeling, and imbalance-handling capabilities.
- 46 new tests; full test suite: 59 passed.
