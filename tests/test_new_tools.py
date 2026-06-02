"""Tests for the new FE operations, EDA analyses, and Modeling tools."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from aiml_discovery.agents.feature_engineering_agent import _apply_operation, _NEW_OPERATIONS


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _num_df(n=50):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.exponential(2, n),
        "c": rng.integers(0, 5, n).astype(float),
        "cat": rng.choice(["X", "Y", "Z"], n),
        "target": rng.integers(0, 2, n),
    })


def _ts_df(n=60):
    dates = pd.date_range("2020-01", periods=n, freq="ME")
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "date": dates,
        "value": np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, 0.1, n),
        "group": ["A"] * n,
    })


def _imbalanced_df(n_maj=100, n_min=10):
    df = pd.concat([
        pd.DataFrame({"a": np.random.randn(n_maj), "b": np.random.randn(n_maj), "y": 0}),
        pd.DataFrame({"a": np.random.randn(n_min) + 3, "b": np.random.randn(n_min), "y": 1}),
    ]).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Feature Engineering — new operations
# ──────────────────────────────────────────────────────────────────────────────

class TestScaling:
    def test_standard_scale(self):
        df = _num_df()
        out, detail = _apply_operation(df, "standard_scale", {})
        assert "StandardScaler" in detail
        assert abs(out["a"].mean()) < 0.1

    def test_minmax_scale(self):
        df = _num_df()
        out, detail = _apply_operation(df, "minmax_scale", {"columns": ["a", "b"]})
        assert out["a"].min() >= 0.0
        assert out["a"].max() <= 1.0

    def test_robust_scale(self):
        df = _num_df()
        out, detail = _apply_operation(df, "robust_scale", {})
        assert out is not None

    def test_max_abs_scale(self):
        df = _num_df()
        out, detail = _apply_operation(df, "max_abs_scale", {})
        assert out["a"].abs().max() <= 1.0 + 1e-9

    def test_power_transform_yeo_johnson(self):
        df = _num_df()
        out, detail = _apply_operation(df, "power_transform", {"method": "yeo-johnson"})
        assert out is not None
        assert "yeo-johnson" in detail

    def test_quantile_transform_normal(self):
        df = _num_df()
        out, detail = _apply_operation(df, "quantile_transform", {"output_distribution": "normal"})
        assert out is not None

    def test_clip_values(self):
        df = _num_df()
        out, detail = _apply_operation(df, "clip_values", {"column": "b", "min": 0, "max": 3})
        assert out["b"].min() >= 0
        assert out["b"].max() <= 3

    def test_winsorize(self):
        df = _num_df()
        out, detail = _apply_operation(df, "winsorize", {"lower": 0.05, "upper": 0.95})
        assert out is not None
        assert "Winsorized" in detail


class TestImputation:
    def test_constant_impute(self):
        df = _num_df()
        df.loc[0, "a"] = np.nan
        out, detail = _apply_operation(df, "constant_impute", {"fill_value": -999})
        assert out["a"].iloc[0] == -999

    def test_knn_impute(self):
        df = _num_df()
        df.loc[:5, "a"] = np.nan
        out, detail = _apply_operation(df, "knn_impute", {"n_neighbors": 3})
        assert out["a"].isna().sum() == 0

    def test_iterative_impute(self):
        df = _num_df()
        df.loc[:3, "a"] = np.nan
        out, detail = _apply_operation(df, "iterative_impute", {"max_iter": 3})
        assert out["a"].isna().sum() == 0

    def test_add_missing_indicators(self):
        df = _num_df()
        df.loc[0, "a"] = np.nan
        out, detail = _apply_operation(df, "add_missing_indicators", {})
        assert "a_was_missing" in out.columns
        assert out["a_was_missing"].iloc[0] == 1


class TestEncoding:
    def test_ordinal_encode(self):
        df = _num_df()
        out, detail = _apply_operation(df, "ordinal_encode", {"columns": ["cat"]})
        assert pd.api.types.is_numeric_dtype(out["cat"])

    def test_label_encode(self):
        df = _num_df()
        out, detail = _apply_operation(df, "label_encode", {"column": "cat"})
        assert pd.api.types.is_numeric_dtype(out["cat"])

    def test_frequency_encode(self):
        df = _num_df()
        out, detail = _apply_operation(df, "frequency_encode", {"columns": ["cat"]})
        assert "cat_freq" in out.columns
        assert out["cat_freq"].between(0, 1).all()

    def test_target_encode(self):
        df = _num_df()
        out, detail = _apply_operation(df, "target_encode", {"columns": ["cat"], "target_column": "target"})
        assert pd.api.types.is_numeric_dtype(out["cat"])

    def test_cyclical_encode(self):
        df = pd.DataFrame({"month": list(range(1, 13)) * 5})
        out, detail = _apply_operation(df, "cyclical_encode", {"column": "month", "period": 12})
        assert "month_sin" in out.columns
        assert "month_cos" in out.columns

    def test_fourier_features(self):
        df = pd.DataFrame({"t": list(range(60))})
        out, detail = _apply_operation(df, "fourier_features", {"column": "t", "period": 12, "order": 2})
        assert "fourier_t_sin1" in out.columns
        assert "fourier_t_cos2" in out.columns

    def test_datetime_parse(self):
        df = pd.DataFrame({"ds": ["2020-01-01", "2020-02-01", "2020-03-01"]})
        out, detail = _apply_operation(df, "datetime_parse", {"columns": ["ds"]})
        assert pd.api.types.is_datetime64_any_dtype(out["ds"])


class TestCleaning:
    def test_drop_constant(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        out, detail = _apply_operation(df, "drop_constant", {})
        assert "a" not in out.columns

    def test_drop_correlated(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0],
                           "b": [1.0, 2.0, 3.0, 4.0, 5.0],  # perfect correlation
                           "c": [1.0, 3.0, 2.0, 5.0, 4.0]})
        out, detail = _apply_operation(df, "drop_correlated", {"threshold": 0.95})
        assert out.shape[1] < df.shape[1]

    def test_drop_constant_no_constants(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        out, detail = _apply_operation(df, "drop_constant", {})
        assert out is None


class TestSelectionReduction:
    def test_select_k_best(self):
        df = _num_df()
        out, detail = _apply_operation(df, "select_k_best", {"target_column": "target", "k": 2})
        assert out is not None
        assert "kept top 2" in detail

    def test_pca(self):
        df = _num_df()
        out, detail = _apply_operation(df, "pca", {"n_components": 2, "target_column": "target"})
        assert "pc_1" in out.columns
        assert "pc_2" in out.columns


class TestOutliers:
    def test_zscore_outlier_removal(self):
        df = _num_df()
        before = len(df)
        out, detail = _apply_operation(df, "zscore_outlier_removal", {"threshold": 2.0})
        assert len(out) <= before

    def test_isolation_forest_outliers(self):
        df = _num_df()
        out, detail = _apply_operation(df, "isolation_forest_outliers", {"contamination": 0.1})
        assert len(out) < len(df)


class TestResampling:
    def test_smote_basic(self):
        df = _imbalanced_df()
        out, detail = _apply_operation(df, "smote", {"target_column": "y"})
        assert out is not None
        counts = out["y"].value_counts()
        assert counts[0] == counts[1]

    def test_random_oversample(self):
        df = _imbalanced_df()
        out, detail = _apply_operation(df, "random_oversample", {"target_column": "y"})
        assert out is not None
        counts = out["y"].value_counts()
        assert counts[0] == counts[1]

    def test_random_undersample(self):
        df = _imbalanced_df()
        out, detail = _apply_operation(df, "random_undersample", {"target_column": "y"})
        assert out is not None
        counts = out["y"].value_counts()
        assert counts[0] == counts[1]

    def test_smote_missing_target_returns_error(self):
        df = _num_df()
        out, detail = _apply_operation(df, "smote", {"target_column": "nonexistent"})
        assert out is None
        assert "required" in detail


# ──────────────────────────────────────────────────────────────────────────────
# EDA analyses
# ──────────────────────────────────────────────────────────────────────────────

from aiml_discovery.agents.eda_agent import _run_analysis, _build_figure


class TestEdaAnalyses:
    def test_class_balance(self):
        df = _imbalanced_df()
        result, fig, title = _run_analysis(df, "test", "class_balance", {"column": "y"})
        assert "class_counts" in result
        assert result["is_imbalanced"] is True
        assert fig is not None

    def test_target_correlation(self):
        df = _num_df()
        result, fig, title = _run_analysis(df, "test", "target_correlation", {"target_column": "target"})
        assert "correlations" in result
        assert fig is not None

    def test_mutual_information(self):
        df = _num_df()
        result, fig, title = _run_analysis(df, "test", "mutual_information", {"target_column": "target"})
        assert "mutual_information" in result

    def test_normality_test(self):
        df = _num_df()
        result, fig, _ = _run_analysis(df, "test", "normality_test", {"column": "a"})
        assert "skewness" in result

    def test_vif(self):
        df = _num_df()
        result, fig, _ = _run_analysis(df, "test", "vif", {})
        assert "vif" in result

    def test_outlier_summary(self):
        df = _num_df()
        result, fig, _ = _run_analysis(df, "test", "outlier_summary", {})
        assert "outlier_summary" in result

    def test_seasonal_decompose(self):
        df = _ts_df()
        result, fig, _ = _run_analysis(df, "test", "seasonal_decompose", {
            "column": "value", "time_column": "date", "period": 12,
        })
        assert "seasonal_strength" in result
        assert fig is not None

    def test_stationarity_test(self):
        df = _ts_df()
        result, fig, _ = _run_analysis(df, "test", "stationarity_test", {"column": "value"})
        assert "adf_stat" in result
        assert "verdict" in result

    def test_acf_pacf(self):
        df = _ts_df()
        result, fig, _ = _run_analysis(df, "test", "acf_pacf", {"column": "value", "nlags": 12})
        assert "acf" in result
        assert "pacf" in result
        assert fig is not None


class TestEdaNewCharts:
    def _df(self):
        df = _ts_df()
        df["date_str"] = df["date"].astype(str)
        return df

    def test_line_chart(self):
        df = self._df()
        fig, title, desc = _build_figure(df, "test", "line", {"x_column": "date", "y_column": "value"})
        assert fig is not None

    def test_qq_plot(self):
        df = _num_df()
        fig, title, desc = _build_figure(df, "test", "qq_plot", {"column": "a"})
        assert fig is not None


# ──────────────────────────────────────────────────────────────────────────────
# Modeling — cross_validate and tune (lightweight smoke tests)
# ──────────────────────────────────────────────────────────────────────────────

from unittest.mock import MagicMock
from aiml_discovery.agents.modeling_agent import ModelingAgent
from aiml_discovery.agents.base import AgentContext


def _make_agent(tmp_path):
    from aiml_discovery.storage import ProjectStore
    store = ProjectStore(tmp_path)
    project_id = store.create_project("test").id
    ctx = AgentContext(project_id=project_id, store=store)
    # Write a CSV dataset
    df = _imbalanced_df(50, 20)
    csv_bytes = df.to_csv(index=False).encode()
    saved = store.save_dataset_file(project_id, "data.csv", csv_bytes)
    store.register_dataset(project_id, name="data", source_name="data.csv",
                           source_type="csv", file_path=str(saved),
                           row_count=len(df), column_count=len(df.columns))
    agent = ModelingAgent(client=MagicMock(), deployment="gpt-test", context=ctx)
    return agent, ctx


def test_cross_validate_model(tmp_path):
    agent, ctx = _make_agent(tmp_path)
    ds = ctx.list_datasets()[0]
    result_json, step, terminate = agent._cross_validate({
        "dataset_id": ds.id,
        "target_column": "y",
        "model_name": "Random Forest",
        "n_splits": 3,
    })
    import json
    result = json.loads(result_json)
    assert "mean_score" in result
    assert "std_score" in result


def test_tune_hyperparameters(tmp_path):
    agent, ctx = _make_agent(tmp_path)
    ds = ctx.list_datasets()[0]
    result_json, step, terminate = agent._tune({
        "dataset_id": ds.id,
        "target_column": "y",
        "model_name": "Random Forest",
        "n_iter": 3,
        "n_splits": 2,
    })
    import json
    result = json.loads(result_json)
    assert "best_score" in result
    assert "best_params" in result


def test_build_ensemble_voting(tmp_path):
    agent, ctx = _make_agent(tmp_path)
    ds = ctx.list_datasets()[0]
    import json
    result_json, step, _ = agent._build_ensemble({
        "dataset_id": ds.id,
        "target_column": "y",
        "model_names": ["Random Forest", "Logistic Regression"],
        "ensemble_type": "voting",
    })
    result = json.loads(result_json)
    assert "metrics" in result
    assert result["ensemble_type"] == "voting"


def test_class_weight_in_train_model():
    from aiml_discovery.training import TrainingSettings, train_automl
    df = _imbalanced_df(80, 20)
    settings = TrainingSettings(
        target_column="y", test_size=0.25,
        class_weight="balanced",
    )
    result, model = train_automl(
        df, settings,
        include_models=["Random Forest", "Logistic Regression"],
    )
    assert result.task_type == "classification"
    assert "f1_weighted" in result.best_metrics


def test_hash_encode():
    df = pd.DataFrame({"id": [f"SKU{i:05d}" for i in range(100)], "val": range(100)})
    out, detail = _apply_operation(df, "hash_encode", {"columns": ["id"], "n_features": 16})
    assert out is not None
    assert "id_hash_0" in out.columns
    assert "id" not in out.columns
    assert "16 features" in detail
