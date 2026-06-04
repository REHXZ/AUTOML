from __future__ import annotations

import pandas as pd

from backend.logic.training import CLASSIFICATION, REGRESSION, TrainingSettings, infer_task_type, train_automl


def test_infer_task_type_for_categorical_target():
    target = pd.Series(["yes", "no", "yes", "no"])

    assert infer_task_type(target) == CLASSIFICATION


def test_infer_task_type_for_continuous_target():
    target = pd.Series([1.1, 2.4, 3.8, 4.2, 5.9, 6.1, 7.7, 8.3, 9.4, 10.2, 11.5, 12.8])

    assert infer_task_type(target) == REGRESSION


def test_train_automl_classification():
    dataframe = pd.DataFrame(
        {
            "age": [22, 25, 29, 35, 42, 51, 55, 61, 48, 39, 33, 27],
            "segment": ["A", "A", "B", "B", "C", "C", "C", "B", "A", "B", "C", "A"],
            "churn": ["no", "no", "yes", "yes", "yes", "no", "no", "yes", "no", "yes", "yes", "no"],
        }
    )

    result, model = train_automl(dataframe, TrainingSettings(target_column="churn", test_size=0.25))

    assert result.task_type == CLASSIFICATION
    assert result.best_model_name
    assert result.best_metrics
    assert hasattr(model, "predict")


def test_train_automl_regression():
    dataframe = pd.DataFrame(
        {
            "tenure": list(range(1, 21)),
            "region": ["north", "south"] * 10,
            "revenue": [value * 12.5 + 3 for value in range(1, 21)],
        }
    )

    result, model = train_automl(dataframe, TrainingSettings(target_column="revenue", test_size=0.25))

    assert result.task_type == REGRESSION
    assert result.best_model_name
    assert "rmse" in result.best_metrics
    assert hasattr(model, "predict")

