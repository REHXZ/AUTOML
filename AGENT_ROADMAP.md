# AI/ML Agent Roadmap & Capability Checklist

> **Living document.** Tracks every capability the AIML Autopilot agents need to
> handle any job under the AI/ML umbrella. Items are ticked off as they land.
>
> Legend: ✅ implemented & tested | 🔜 planned | ⬜ not yet planned

---

## CRISP-DM Lifecycle Coverage

| Phase | Capability | Status |
|-------|-----------|--------|
| Business Understanding | Scientist frames problem, asks targeted clarifying questions | ✅ |
| Data Understanding | EDA Agent profiles, charts, runs statistical analyses | ✅ |
| Data Preparation | Feature Engineering Agent applies 50+ transforms | ✅ |
| Modeling | AutoML baseline (25+ models), chronological splits, ensembles | ✅ |
| Evaluation | Review Agent critiques leakage, imbalance, underfitting | ✅ |
| Iteration | Fine Tuning Agent acts on review; loops until metric plateau | ✅ |
| Deployment / Serving | Score new data against a saved model via API | ✅ |
| Monitoring | Detect feature/label drift between reference and live data | ✅ |

---

## 1. Feature Engineering Agent — Operations

### 1.1 Cleaning & Row Operations
| Operation | Status |
|-----------|--------|
| `drop_high_missing` — drop columns above a missing-fraction threshold | ✅ |
| `drop_duplicates` — remove duplicate rows | ✅ |
| `select_columns` — keep only listed columns | ✅ |
| `drop_columns` — remove listed columns | ✅ |
| `filter_outliers` — IQR (1.5×) row removal | ✅ |
| `drop_constant` — drop zero-variance columns | ✅ |
| `drop_correlated` — drop one of each highly-correlated pair | ✅ |

### 1.2 Imputation
| Operation | Status |
|-----------|--------|
| `impute_missing` — median / mode fill | ✅ |
| `constant_impute` — fill with a fixed value | ✅ |
| `knn_impute` — KNNImputer | ✅ |
| `iterative_impute` — MICE / IterativeImputer | ✅ |
| `add_missing_indicators` — add `{col}_was_missing` flags | ✅ |

### 1.3 Scaling & Numeric Transforms
| Operation | Status |
|-----------|--------|
| `log_transform` — log1p of chosen columns (adds new cols) | ✅ |
| `target_log_transform` — log1p of a skewed regression target in-place | ✅ |
| `standard_scale` — StandardScaler (z-score) | ✅ |
| `minmax_scale` — MinMaxScaler to [0, 1] | ✅ |
| `robust_scale` — RobustScaler (median/IQR) | ✅ |
| `max_abs_scale` — MaxAbsScaler (sparse-friendly) | ✅ |
| `power_transform` — Yeo-Johnson / Box-Cox | ✅ |
| `quantile_transform` — rank-based uniform/normal | ✅ |
| `bin_numeric` — quantile-bin a numeric column | ✅ |
| `clip_values` — clip a column to [min, max] | ✅ |
| `winsorize` — clip to percentile range | ✅ |

### 1.4 Encoding
| Operation | Status |
|-----------|--------|
| `one_hot_encode` — one-hot low-cardinality categoricals | ✅ |
| `ordinal_encode` — OrdinalEncoder integer codes | ✅ |
| `label_encode` — integer-encode a single column | ✅ |
| `frequency_encode` — replace categories with frequency | ✅ |
| `target_encode` — smoothed mean-of-target encoding | ✅ |
| `hash_encode` — feature hashing for very high cardinality | ✅ |
| `encode_dates` — expand datetime → year/month/day/dow/quarter | ✅ |
| `datetime_parse` — coerce columns to datetime dtype | ✅ |
| `cyclical_encode` — sin/cos for cyclic features (month, hour, dow) | ✅ |
| `fourier_features` — sin/cos harmonics for seasonal columns | ✅ |
| `interaction_features` — multiplicative pairs a×b | ✅ |
| `polynomial_features` — squared/cubed terms | ✅ |
| `rename_columns` — rename columns | ✅ |

### 1.5 Aggregation
| Operation | Status |
|-----------|--------|
| `groupby_aggregate` — group-by rollups (sum/mean/nunique/count/min/max/first/last) | ✅ |

### 1.6 Time-Series Features (Forecasting)
| Operation | Status |
|-----------|--------|
| `dense_panel` — fill missing (group × time) combinations | ✅ |
| `create_lag_features` — backward lags within group | ✅ |
| `create_rolling_features` — leakage-safe rolling-window aggregates | ✅ |
| `create_lead_target` — forward-shifted forecasting targets (t+1, t+3 …) | ✅ |

### 1.7 Feature Selection & Dimensionality Reduction
| Operation | Status |
|-----------|--------|
| `select_k_best` — SelectKBest (F-test or mutual information) | ✅ |
| `select_from_model` — SelectFromModel (RandomForest importance) | ✅ |
| `rfe_select` — Recursive Feature Elimination | ✅ |
| `pca` — PCA dimensionality reduction | ✅ |

### 1.8 Outlier Handling
| Operation | Status |
|-----------|--------|
| `filter_outliers` — IQR row removal | ✅ |
| `winsorize` — clip to percentile range | ✅ |
| `clip_values` — clip to explicit [min, max] | ✅ |
| `zscore_outlier_removal` — drop rows with \|z\| above threshold | ✅ |
| `isolation_forest_outliers` — IsolationForest anomaly removal | ✅ |

### 1.9 Class-Imbalance Resampling
| Operation | Status |
|-----------|--------|
| `smote` — SMOTE (auto-SMOTENC for mixed data) | ✅ |
| `borderline_smote` — Borderline-SMOTE | ✅ |
| `adasyn` — ADASYN adaptive oversampling | ✅ |
| `random_oversample` — random minority oversampling | ✅ |
| `random_undersample` — random majority undersampling | ✅ |
| `smote_tomek` — SMOTE + Tomek-link cleaning | ✅ |
| `smote_enn` — SMOTE + Edited-Nearest-Neighbours | ✅ |

### 1.10 NLP / Text Operations
| Operation | Status |
|-----------|--------|
| `text_clean` — lowercase, strip punctuation/digits/whitespace | ✅ |
| `text_length_features` — char count, word count, avg word length | ✅ |
| `tfidf_vectorize` — TF-IDF sparse → dense top-N features | ✅ |
| `count_vectorize` — Bag-of-words count matrix top-N features | ✅ |
| `lsa_text` — Latent Semantic Analysis (TF-IDF + TruncatedSVD) | ✅ |

### 1.11 Escape Hatch
| Operation | Status |
|-----------|--------|
| `execute_python` — run arbitrary pandas/numpy on the dataframe | ✅ |

---

## 2. EDA Agent

### 2.1 Charts
| Chart Type | Status |
|-----------|--------|
| `histogram` — single-column distribution | ✅ |
| `bar` — top-N value counts | ✅ |
| `scatter` — x vs y (optional colour) | ✅ |
| `correlation_heatmap` — numeric correlation matrix | ✅ |
| `box` — box plot (optional group-by) | ✅ |
| `violin` — violin plot (optional group-by) | ✅ |
| `pairplot` — scatter matrix of up to 4 numerics | ✅ |
| `missing_heatmap` — missing-value pattern | ✅ |
| `line` — value(s) over an ordered/time axis | ✅ |
| `qq_plot` — Normal Q-Q plot | ✅ |

### 2.2 Statistical Analyses
| Analysis Type | Status |
|--------------|--------|
| `class_balance` — target class counts + imbalance ratio | ✅ |
| `target_correlation` — feature↔target Pearson correlation | ✅ |
| `mutual_information` — non-linear feature-target MI scores | ✅ |
| `normality_test` — Shapiro / D'Agostino + skew & kurtosis | ✅ |
| `vif` — Variance Inflation Factor (multicollinearity) | ✅ |
| `outlier_summary` — IQR / z-score outlier counts per column | ✅ |
| `seasonal_decompose` — trend / seasonal / residual decomposition | ✅ |
| `stationarity_test` — ADF + KPSS stationarity tests | ✅ |
| `acf_pacf` — autocorrelation & partial-autocorrelation with chart | ✅ |
| `ttest` — independent / paired Student's t-test between two groups | ✅ |
| `chi2_test` — chi-squared test of independence for categoricals | ✅ |
| `anova` — one-way ANOVA across 3+ groups | ✅ |
| `mannwhitney` — Mann-Whitney U (non-parametric alternative to t-test) | ✅ |
| `kruskal_wallis` — Kruskal-Wallis H (non-parametric ANOVA) | ✅ |
| `correlation_significance` — Pearson r with p-values for all numeric pairs | ✅ |

---

## 3. Modeling Agent

### 3.1 AutoML Catalogue
| Category | Models | Status |
|----------|--------|--------|
| Classification | Baseline, Logistic Regression, SGD, Linear SVC, Gaussian NB, Bernoulli NB, LDA, QDA, Decision Tree, Extra Trees, Random Forest, AdaBoost, Gradient Boosting, Hist Gradient Boosting, KNN, SVC (RBF), MLP, Bagging, XGBoost*, LightGBM*, CatBoost* | ✅ |
| Regression | Baseline, Linear Regression, Ridge, Lasso, ElasticNet, Bayesian Ridge, Huber, SGD Regressor, Decision Tree, Extra Trees, Random Forest, AdaBoost, Gradient Boosting, Hist Gradient Boosting, KNN, Linear SVR, SVR (RBF), MLP, Bagging, XGBoost*, LightGBM*, CatBoost* | ✅ |

`*` optional; auto-included when library is installed.

### 3.2 Training & Validation
| Capability | Status |
|-----------|--------|
| Chronological split (`time_column`) — honest forecasting backtest | ✅ |
| `include_models` / `custom_models` — restrict / extend catalogue | ✅ |
| `class_weight="balanced"` — cost-sensitive imbalanced classification | ✅ |
| `cross_validate_model` — k-fold / TimeSeriesSplit CV with mean ± std | ✅ |
| `tune_hyperparameters` — RandomizedSearchCV over 13 model families | ✅ |
| `build_ensemble` — Voting or Stacking over selected models | ✅ |
| `tune_hyperparameters_optuna` — Bayesian HPO via Optuna | ✅ |

### 3.3 Diagnostics & Visualisation
| Capability | Status |
|-----------|--------|
| `predicted_vs_actual` — regression scatter with y=x reference | ✅ |
| `forecast` — actual vs predicted lines in time order | ✅ |
| `residuals` — residual scatter plot | ✅ |
| `residuals_over_time` — residuals plotted in time order | ✅ |
| `confusion_matrix` — classification confusion matrix | ✅ |
| `feature_importance` — top-N importances (tree models) | ✅ |
| `leaderboard` — primary metric per candidate model in run | ✅ |
| `compare_runs` — side-by-side bar chart across experiments | ✅ |
| `explain_model` — SHAP Shapley value summary plot | ✅ |

### 3.4 Time-Series Dedicated Models
| Capability | Status |
|-----------|--------|
| `train_arima` — ARIMA/SARIMA via statsmodels (AIC grid-search order selection) | ✅ |
| `arima_forecast` — n-step-ahead forecast with 95% confidence bands | ✅ |

---

## 4. Drift Detection Agent (new)

| Capability | Status |
|-----------|--------|
| `compare_distributions` — PSI, KS-test, Jensen-Shannon divergence per feature | ✅ |
| `run_drift_report` — full reference vs current dataset drift report | ✅ |
| Scientist can delegate to Drift Agent via `delegate_to_drift_detection` | ✅ |

---

## 5. Scoring / Inference (API)

| Capability | Status |
|-----------|--------|
| `POST /api/projects/{id}/runs/{run_id}/score` — score new rows with saved model | ✅ |
| JSON body `{"data": [...rows...]}` or multipart CSV file upload | ✅ |
| Returns `{"predictions": [...], "run_id": "...", "task_type": "..."}` | ✅ |

---

## 6. Orchestration & Research

| Agent | Capability | Status |
|-------|-----------|--------|
| Scientist | Phase management, `ask_user`, delegation, `finalize_strategy` | ✅ |
| Scientist | Delegate to Drift Detection Agent | ✅ |
| Review | Critique runs, spawn researcher, rank improvements | ✅ |
| Fine Tuning | Spawn FE + Modeling sub-agents, iterate | ✅ |
| Researcher | `search_web` + `fetch_page` via SearXNG | ✅ |

---

## 7. Infrastructure & Export

| Capability | Status |
|-----------|--------|
| Session persistence (JSONL, resume after refresh) | ✅ |
| Jupyter notebook export (CRISP-DM structured, runnable Plotly code) | ✅ |
| React dashboard (swimlane graph, timeline, SSE stream) | ✅ |
| Streamlit manual workspace | ✅ |
| Docker SearXNG for web research | ✅ |
| OpenTelemetry tracing (Arize Phoenix) | ✅ |

---

## 8. Known Gaps / Future Work

| Item | Priority | Notes |
|------|----------|-------|
| Deep learning integration (PyTorch/TF) | Low | Too heavy for local-first; use custom_models escape hatch |
| Synthetic data generation (CTGAN) | Low | Needs `sdv` package (~500 MB) |
| Causal inference (DoWhy/EconML) | Low | Specialist use case |
| Federated learning | Very Low | Requires distributed infrastructure |
| LLM/RAG pipeline builder | Future | Agentic workflow for text-heavy tasks |

---

*Last updated: 2026-06-02 — branch `claude/intelligent-sagan-G02fc`*
