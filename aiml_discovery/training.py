from __future__ import annotations

import importlib
import inspect
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .logging_setup import configure_logging

configure_logging()
log = logging.getLogger(__name__)

CLASSIFICATION = "classification"
REGRESSION = "regression"


@dataclass(frozen=True)
class TrainingSettings:
    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    max_rows: int | None = None
    # When set the train/test split is CHRONOLOGICAL — rows are sorted by
    # this column and the last `test_size` fraction becomes the test set.
    time_column: str | None = None


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    trained_at: str
    task_type: str
    target_column: str
    row_count: int
    feature_columns: list[str]
    best_model_name: str
    best_metrics: dict[str, float]
    leaderboard: list[dict[str, Any]]
    settings: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def infer_task_type(target: pd.Series) -> str:
    clean_target = target.dropna()
    unique_count = int(clean_target.nunique(dropna=True))

    if unique_count <= 1:
        return CLASSIFICATION
    if pd.api.types.is_bool_dtype(clean_target):
        return CLASSIFICATION
    if not pd.api.types.is_numeric_dtype(clean_target):
        return CLASSIFICATION
    if pd.api.types.is_integer_dtype(clean_target) and unique_count <= 20:
        return CLASSIFICATION

    distinct_ratio = unique_count / max(len(clean_target), 1)
    if unique_count <= 10 and distinct_ratio <= 0.2:
        return CLASSIFICATION
    return REGRESSION


def list_available_models(task_type: str) -> list[str]:
    """Return names of all standard models for a task type (used by agent UI)."""
    return list(_candidate_models(task_type, random_state=42, n_jobs=1).keys())


def _cpu_limit() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


# ─────────────────────────────────────────────────────────────────────────────
# Optional library detection
# ─────────────────────────────────────────────────────────────────────────────


def _optional_models(task_type: str, random_state: int, n_jobs: int) -> dict[str, Any]:
    """Try to import XGBoost, LightGBM, CatBoost and return available models."""
    extras: dict[str, Any] = {}

    try:
        import xgboost as xgb
        if task_type == REGRESSION:
            extras["XGBoost"] = xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                random_state=random_state, n_jobs=n_jobs,
                tree_method="hist", verbosity=0,
            )
        else:
            extras["XGBoost"] = xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                random_state=random_state, n_jobs=n_jobs,
                tree_method="hist", verbosity=0, eval_metric="logloss",
            )
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        if task_type == REGRESSION:
            extras["LightGBM"] = lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                random_state=random_state, n_jobs=n_jobs, verbose=-1,
            )
        else:
            extras["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                random_state=random_state, n_jobs=n_jobs, verbose=-1,
            )
    except ImportError:
        pass

    try:
        import catboost as cb
        if task_type == REGRESSION:
            extras["CatBoost"] = cb.CatBoostRegressor(
                iterations=200, learning_rate=0.05, depth=6,
                random_seed=random_state, verbose=0,
            )
        else:
            extras["CatBoost"] = cb.CatBoostClassifier(
                iterations=200, learning_rate=0.05, depth=6,
                random_seed=random_state, verbose=0,
            )
    except ImportError:
        pass

    return extras


# ─────────────────────────────────────────────────────────────────────────────
# Custom model instantiation
# ─────────────────────────────────────────────────────────────────────────────


def _instantiate_custom_model(spec: dict) -> tuple[str, Any] | None:
    """Instantiate a user-supplied model from a spec dict.

    spec format:
        {
            "name": "My XGBoost",               # display name (optional)
            "class": "xgboost.XGBRegressor",    # dotted import path (required)
            "params": {"n_estimators": 300}     # constructor kwargs (optional)
        }

    Returns (name, instance) or None on failure.
    """
    class_path = (spec.get("class") or "").strip()
    if not class_path or "." not in class_path:
        log.warning("custom model spec missing valid 'class' field: %s", spec)
        return None
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        params = spec.get("params") or {}
        instance = cls(**params)
        name = (spec.get("name") or class_name).strip()
        log.info("custom model instantiated: %s → %s", name, class_path)
        return name, instance
    except Exception as exc:
        log.warning("custom model '%s' failed to instantiate: %s", class_path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Model catalogue
# ─────────────────────────────────────────────────────────────────────────────


def _candidate_models(
    task_type: str,
    random_state: int,
    n_jobs: int,
    include_models: list[str] | None = None,
    custom_models: list[dict] | None = None,
) -> dict[str, Any]:
    """Return ordered dict of name → unfitted model instance.

    include_models: if provided, only standard models whose name appears in
        this list are included. Custom models are always appended.
    custom_models: list of {"name", "class", "params"} specs. Appended after
        standard models. Failures are logged and skipped.
    """
    if task_type == CLASSIFICATION:
        standard: dict[str, Any] = {
            # ── Baselines ──────────────────────────────────────────────
            "Baseline (Majority)": DummyClassifier(strategy="most_frequent"),
            # ── Linear ────────────────────────────────────────────────
            "Logistic Regression": LogisticRegression(
                max_iter=1_000, random_state=random_state,
            ),
            "SGD Classifier": SGDClassifier(
                max_iter=1_000, random_state=random_state, n_jobs=n_jobs,
            ),
            "Linear SVC": CalibratedClassifierCV(
                LinearSVC(max_iter=3_000, random_state=random_state), cv=3,
            ),
            # ── Probabilistic ─────────────────────────────────────────
            "Gaussian Naive Bayes": GaussianNB(),
            "Bernoulli Naive Bayes": BernoulliNB(),
            # ── Discriminant Analysis ─────────────────────────────────
            "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
            "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
            # ── Trees ─────────────────────────────────────────────────
            "Decision Tree": DecisionTreeClassifier(
                max_depth=8, random_state=random_state,
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=120, min_samples_leaf=2,
                random_state=random_state, n_jobs=n_jobs,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=120, min_samples_leaf=2,
                random_state=random_state, n_jobs=n_jobs,
            ),
            # ── Boosting ──────────────────────────────────────────────
            "AdaBoost": AdaBoostClassifier(
                n_estimators=100, random_state=random_state,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=120, random_state=random_state,
            ),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(
                max_iter=200, random_state=random_state,
            ),
            # ── Instance-based ────────────────────────────────────────
            "K-Nearest Neighbors": KNeighborsClassifier(
                n_neighbors=5, n_jobs=n_jobs,
            ),
            # ── SVM ───────────────────────────────────────────────────
            "SVC (RBF)": SVC(kernel="rbf", C=1.0, probability=True),
            # ── Neural Network ────────────────────────────────────────
            "MLP": MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500,
                random_state=random_state,
            ),
            # ── Bagging ───────────────────────────────────────────────
            "Bagging": BaggingClassifier(
                n_estimators=20, random_state=random_state, n_jobs=n_jobs,
            ),
        }
    else:
        standard = {
            # ── Baselines ──────────────────────────────────────────────
            "Baseline (Mean)": DummyRegressor(strategy="mean"),
            # ── Linear ────────────────────────────────────────────────
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(alpha=0.01, max_iter=3_000),
            "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=3_000),
            "Bayesian Ridge": BayesianRidge(),
            "Huber Regressor": HuberRegressor(max_iter=300),
            "SGD Regressor": SGDRegressor(max_iter=1_000, random_state=random_state),
            # ── Trees ─────────────────────────────────────────────────
            "Decision Tree": DecisionTreeRegressor(
                max_depth=8, random_state=random_state,
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=120, min_samples_leaf=2,
                random_state=random_state, n_jobs=n_jobs,
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=120, min_samples_leaf=2,
                random_state=random_state, n_jobs=n_jobs,
            ),
            # ── Boosting ──────────────────────────────────────────────
            "AdaBoost": AdaBoostRegressor(
                n_estimators=100, random_state=random_state,
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=120, random_state=random_state,
            ),
            "Hist Gradient Boosting": HistGradientBoostingRegressor(
                max_iter=200, random_state=random_state,
            ),
            # ── Instance-based ────────────────────────────────────────
            "K-Nearest Neighbors": KNeighborsRegressor(
                n_neighbors=5, n_jobs=n_jobs,
            ),
            # ── SVM ───────────────────────────────────────────────────
            "Linear SVR": LinearSVR(max_iter=3_000, random_state=random_state),
            "SVR (RBF)": SVR(kernel="rbf", C=1.0),
            # ── Neural Network ────────────────────────────────────────
            "MLP": MLPRegressor(
                hidden_layer_sizes=(100, 50), max_iter=500,
                random_state=random_state,
            ),
            # ── Bagging ───────────────────────────────────────────────
            "Bagging": BaggingRegressor(
                n_estimators=20, random_state=random_state, n_jobs=n_jobs,
            ),
        }

    # Apply include_models filter to standard models
    if include_models:
        include_set = set(include_models)
        standard = {k: v for k, v in standard.items() if k in include_set}

    # Add optional third-party models (XGBoost, LightGBM, CatBoost)
    # These are skipped silently if the library is not installed.
    if not include_models:
        # Only add optional models when running the full catalogue.
        standard.update(_optional_models(task_type, random_state, n_jobs))
    else:
        # Add optional models if explicitly requested by name.
        opt = _optional_models(task_type, random_state, n_jobs)
        for name in include_models:
            if name in opt and name not in standard:
                standard[name] = opt[name]

    # Append custom models (always, regardless of include_models).
    if custom_models:
        for spec in custom_models:
            result = _instantiate_custom_model(spec)
            if result is not None:
                name, model = result
                standard[name] = model

    return standard


# ─────────────────────────────────────────────────────────────────────────────
# Training stream
# ─────────────────────────────────────────────────────────────────────────────


def train_automl_stream(
    dataframe: pd.DataFrame,
    settings: TrainingSettings,
    custom_models: list[dict] | None = None,
    include_models: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Generator yielding real-time training progress dicts, then a final 'done' event.

    Event types:
      model_start  – fired before each model begins fitting
      epoch        – fired after each warm-start step for iterative models
      model_done   – fired after each model finishes (success or failed)
      done         – final event carrying 'result' (TrainingResult) and 'pipeline'

    custom_models: list of {"name", "class", "params"} specs appended to the catalogue.
    include_models: if set, only standard models in this list are run (plus custom).
    """
    n_jobs = _cpu_limit()
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_var] = str(n_jobs)

    if settings.target_column not in dataframe.columns:
        raise ValueError(f"Target column '{settings.target_column}' was not found.")

    working = dataframe.dropna(subset=[settings.target_column]).copy()
    if settings.max_rows and len(working) > settings.max_rows:
        working = working.sample(n=settings.max_rows, random_state=settings.random_state)

    if len(working) < 5:
        raise ValueError("At least 5 rows with a target value are required for training.")

    feature_columns = [
        column for column in working.columns
        if column != settings.target_column
        and column != settings.time_column
    ]
    if not feature_columns:
        raise ValueError("Training requires at least one feature column besides the target.")

    target = working[settings.target_column]
    task_type = infer_task_type(target)

    features = working[feature_columns]
    test_time_values: pd.Series | None = None

    if settings.time_column and settings.time_column in working.columns:
        order = pd.to_datetime(working[settings.time_column], errors="coerce")
        if order.isna().all():
            order = working[settings.time_column]
        sort_idx = order.argsort(kind="mergesort")
        features = features.iloc[sort_idx].reset_index(drop=True)
        target = target.iloc[sort_idx].reset_index(drop=True)
        time_aligned = working[settings.time_column].iloc[sort_idx].reset_index(drop=True)
        split_at = int(len(features) * (1.0 - settings.test_size))
        if split_at < 2 or split_at >= len(features):
            raise ValueError(
                f"Chronological split with test_size={settings.test_size} produced "
                f"unusable train ({split_at}) / test ({len(features) - split_at}) sizes."
            )
        x_train = features.iloc[:split_at]
        x_test = features.iloc[split_at:]
        y_train = target.iloc[:split_at]
        y_test = target.iloc[split_at:]
        test_time_values = time_aligned.iloc[split_at:].reset_index(drop=True)
    else:
        stratify = _stratify_target(target, settings.test_size) if task_type == CLASSIFICATION else None
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=settings.test_size,
            random_state=settings.random_state,
            stratify=stratify,
        )

    preprocessor = build_preprocessor(features)
    candidates = _candidate_models(
        task_type, settings.random_state, n_jobs,
        include_models=include_models,
        custom_models=custom_models,
    )
    leaderboard: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    predictions_by_model: dict[str, Any] = {}
    total_models = len(candidates)

    log.info(
        "train_automl_stream | task=%s models=%d include=%s custom=%d",
        task_type, total_models,
        include_models or "all",
        len(custom_models) if custom_models else 0,
    )

    for model_idx, (model_name, model) in enumerate(candidates.items(), 1):
        yield {
            "type": "model_start",
            "model_name": model_name,
            "model_idx": model_idx,
            "total_models": total_models,
        }

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        try:
            if isinstance(getattr(model, "n_estimators", None), int):
                total_estimators = model.n_estimators
                num_steps = 10
                step_size = max(1, total_estimators // num_steps)
                model.warm_start = True

                for step in range(1, num_steps + 1):
                    model.n_estimators = min(step * step_size, total_estimators)
                    pipeline.fit(x_train, y_train)
                    yield {
                        "type": "epoch",
                        "model_name": model_name,
                        "model_idx": model_idx,
                        "total_models": total_models,
                        "epoch": step,
                        "total_epochs": num_steps,
                        "estimators": model.n_estimators,
                        "total_estimators": total_estimators,
                    }

                if model.n_estimators < total_estimators:
                    model.n_estimators = total_estimators
                    pipeline.fit(x_train, y_train)
            else:
                pipeline.fit(x_train, y_train)

            predictions = pipeline.predict(x_test)
            metrics = _evaluate_model(task_type, y_test, predictions, pipeline, x_test)
            leaderboard.append({"model": model_name, "status": "success", "metrics": metrics, "error": ""})
            fitted_models[model_name] = pipeline
            predictions_by_model[model_name] = predictions
            yield {
                "type": "model_done",
                "model_name": model_name,
                "model_idx": model_idx,
                "total_models": total_models,
                "status": "success",
                "metrics": metrics,
            }
        except Exception as exc:
            log.warning("model '%s' failed: %s", model_name, exc)
            leaderboard.append({"model": model_name, "status": "failed", "metrics": {}, "error": str(exc)})
            yield {
                "type": "model_done",
                "model_name": model_name,
                "model_idx": model_idx,
                "total_models": total_models,
                "status": "failed",
                "error": str(exc),
            }

    leaderboard = _rank_leaderboard(task_type, leaderboard)
    successes = [entry for entry in leaderboard if entry["status"] == "success"]
    if not successes:
        errors = "; ".join(f"{entry['model']}: {entry['error']}" for entry in leaderboard)
        raise ValueError(f"No candidate model could be trained. {errors}")

    best_entry = successes[0]
    best_model_name = best_entry["model"]
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")

    diagnostics: dict[str, Any] = {}
    best_predictions = predictions_by_model.get(best_model_name)
    if best_predictions is not None:
        try:
            from .diagnostics import build_diagnostics_dict
            diagnostics = build_diagnostics_dict(
                task_type, y_test, best_predictions, x_test,
                time_values=test_time_values,
            )
        except Exception:
            diagnostics = {}

    result = TrainingResult(
        run_id=run_id,
        trained_at=datetime.now(timezone.utc).isoformat(),
        task_type=task_type,
        target_column=settings.target_column,
        row_count=int(len(working)),
        feature_columns=[str(column) for column in feature_columns],
        best_model_name=best_model_name,
        best_metrics=best_entry["metrics"],
        leaderboard=leaderboard,
        settings=asdict(settings),
        diagnostics=diagnostics,
    )
    yield {"type": "done", "result": result, "pipeline": fitted_models[best_model_name]}


def train_automl(
    dataframe: pd.DataFrame,
    settings: TrainingSettings,
    custom_models: list[dict] | None = None,
    include_models: list[str] | None = None,
) -> tuple[TrainingResult, Pipeline]:
    for update in train_automl_stream(dataframe, settings, custom_models=custom_models, include_models=include_models):
        if update["type"] == "done":
            return update["result"], update["pipeline"]
    raise RuntimeError("Training stream ended without a result.")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────


def build_preprocessor(dataframe: pd.DataFrame) -> ColumnTransformer:
    numeric_features = dataframe.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in dataframe.columns if column not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline(steps=[
                ("imputer", _simple_imputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline(steps=[
                ("imputer", _simple_imputer(strategy="most_frequent")),
                ("encoder", _one_hot_encoder()),
            ]),
            categorical_features,
        ))

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics & ranking
# ─────────────────────────────────────────────────────────────────────────────


def _evaluate_model(
    task_type: str,
    y_test: pd.Series,
    predictions: Any,
    pipeline: Pipeline,
    x_test: pd.DataFrame,
) -> dict[str, float]:
    if task_type == CLASSIFICATION:
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            "precision_weighted": float(
                precision_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        }
        if hasattr(pipeline, "predict_proba") and y_test.nunique(dropna=True) == 2:
            try:
                probabilities = pipeline.predict_proba(x_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
            except Exception:
                pass
        return metrics

    mse = mean_squared_error(y_test, predictions)
    return {
        "r2": float(r2_score(y_test, predictions)),
        "rmse": float(mse**0.5),
        "mae": float(mean_absolute_error(y_test, predictions)),
    }


def _rank_leaderboard(task_type: str, leaderboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successes = [entry for entry in leaderboard if entry["status"] == "success"]
    failures = [entry for entry in leaderboard if entry["status"] != "success"]

    if task_type == CLASSIFICATION:
        successes.sort(
            key=lambda entry: (
                entry["metrics"].get("f1_weighted", float("-inf")),
                entry["metrics"].get("accuracy", float("-inf")),
            ),
            reverse=True,
        )
    else:
        successes.sort(
            key=lambda entry: (
                entry["metrics"].get("r2", float("-inf")),
                -entry["metrics"].get("rmse", float("inf")),
            ),
            reverse=True,
        )

    ranked = successes + failures
    for index, entry in enumerate(ranked, start=1):
        entry["rank"] = index if entry["status"] == "success" else None
    return ranked


def _stratify_target(target: pd.Series, test_size: float) -> pd.Series | None:
    counts = target.value_counts()
    if len(counts) <= 1 or counts.min() < 2:
        return None

    test_count = max(1, round(len(target) * test_size))
    train_count = len(target) - test_count
    if test_count < len(counts) or train_count < len(counts):
        return None
    return target


def _simple_imputer(strategy: str) -> SimpleImputer:
    parameters = {"strategy": strategy}
    if "keep_empty_features" in inspect.signature(SimpleImputer).parameters:
        parameters["keep_empty_features"] = True
    return SimpleImputer(**parameters)


def _one_hot_encoder() -> OneHotEncoder:
    parameters: dict[str, Any] = {"handle_unknown": "ignore"}
    signature = inspect.signature(OneHotEncoder)
    if "sparse_output" in signature.parameters:
        parameters["sparse_output"] = False
    else:
        parameters["sparse"] = False
    return OneHotEncoder(**parameters)
