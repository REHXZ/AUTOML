from __future__ import annotations

import inspect
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLASSIFICATION = "classification"
REGRESSION = "regression"


@dataclass(frozen=True)
class TrainingSettings:
    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    max_rows: int | None = None
    # When set, the train/test split is CHRONOLOGICAL — rows are sorted by
    # this column and the last `test_size` fraction becomes the test set.
    # Use this for forecasting / time-series tasks to avoid leakage from a
    # random shuffle. The column must exist in the dataframe.
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


def _cpu_limit() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


def train_automl_stream(
    dataframe: pd.DataFrame, settings: TrainingSettings
) -> Iterator[dict[str, Any]]:
    """Generator yielding real-time training progress dicts, then a final 'done' event.

    Event types:
      model_start  – fired before each model begins fitting
      epoch        – fired after each warm-start step for iterative models
      model_done   – fired after each model finishes (success or failed)
      done         – final event carrying 'result' (TrainingResult) and 'pipeline'
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

    # Exclude the target AND the time column (when used for chronological split)
    # from the feature set. The time column is only used to ORDER the data; using
    # it as a feature would let the model memorise the date and would also break
    # sklearn's SimpleImputer (which doesn't accept datetime64 dtype).
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
        # Chronological holdout: sort by time, last test_size fraction = test set.
        order = pd.to_datetime(working[settings.time_column], errors="coerce")
        if order.isna().all():
            order = working[settings.time_column]
        sort_idx = order.argsort(kind="mergesort")
        features = features.iloc[sort_idx].reset_index(drop=True)
        target = target.iloc[sort_idx].reset_index(drop=True)
        # Keep the time column aligned for diagnostics (since we dropped it from features).
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
    candidates = _candidate_models(task_type, settings.random_state, n_jobs)
    leaderboard: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    predictions_by_model: dict[str, Any] = {}
    total_models = len(candidates)

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
        except Exception as exc:  # pragma: no cover - surfaced in UI and metadata.
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

    # Build diagnostic data (y_test / y_pred / optional time axis) from the best model.
    diagnostics: dict[str, Any] = {}
    best_predictions = predictions_by_model.get(best_model_name)
    if best_predictions is not None:
        try:
            from .diagnostics import build_diagnostics_dict

            diagnostics = build_diagnostics_dict(
                task_type, y_test, best_predictions, x_test,
                time_values=test_time_values,
            )
        except Exception:  # pragma: no cover - diagnostics must never break a run.
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


def train_automl(dataframe: pd.DataFrame, settings: TrainingSettings) -> tuple[TrainingResult, Pipeline]:
    for update in train_automl_stream(dataframe, settings):
        if update["type"] == "done":
            return update["result"], update["pipeline"]
    raise RuntimeError("Training stream ended without a result.")


def build_preprocessor(dataframe: pd.DataFrame) -> ColumnTransformer:
    numeric_features = dataframe.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in dataframe.columns if column not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", _simple_imputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", _simple_imputer(strategy="most_frequent")),
                        ("encoder", _one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _candidate_models(task_type: str, random_state: int, n_jobs: int = 1) -> dict[str, Any]:
    if task_type == CLASSIFICATION:
        return {
            "Baseline Majority Class": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": LogisticRegression(max_iter=1_000),
            "Random Forest": RandomForestClassifier(
                n_estimators=120,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=n_jobs,
            ),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        }

    return {
        "Baseline Mean": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }


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
            probabilities = pipeline.predict_proba(x_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
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
