"""AI autopilot: uses OpenAI function calling to autonomously analyse data and run AutoML."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .ingestion import load_dataset
from .profiling import profile_dataframe
from .reporting import build_markdown_report
from .storage import DatasetInfo, ProjectStore
from .training import TrainingSettings, train_automl


@dataclass
class AutopilotStep:
    index: int
    kind: str  # "thought"|"tool_call"|"tool_result"|"chart"|"new_dataset"|"training"|"summary"
    title: str
    detail: str = ""
    data: dict[str, Any] | None = None


_SYSTEM_PROMPT = """\
You are an expert ML analyst with full autonomous control over an AutoML discovery platform.
You have vision capability — when you create a chart you will receive the image and can describe
what you observe in it to inform your strategy.

Your mission — execute step by step using the tools provided:
1. Call list_datasets to see all available datasets.
2. For EVERY dataset: call profile_dataset to understand structure, quality, and ML potential.
3. During EDA, call create_chart to visualise the data. For each dataset you should create:
   - A correlation_heatmap (if 2+ numeric columns exist).
   - A histogram for each numeric column that looks like a potential target.
   - A bar chart for each categorical column with low cardinality (≤20 unique values).
   - A missing_heatmap if any missing values exist.
   - A scatter plot between strong numeric correlates when the correlation heatmap reveals them.
   After each chart examine the image you receive and describe what you observe.
4. Based on profiling and charts, decide which datasets and target columns are worth modelling.
5. Optionally call create_derived_dataset to produce cleaner/engineered datasets
   (e.g. drop high-missing columns, remove outliers, encode dates).
6. Call train_model for the best dataset+target pair(s).
7. Call finalize_strategy with a comprehensive markdown report covering:
   your EDA findings (cite chart observations), derived datasets created, training results,
   and specific actionable recommendations.

Be systematic. Profile and chart before acting. Describe what you see in each chart.
"""

_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_datasets",
            "description": "List all datasets registered in the current project.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "profile_dataset",
            "description": (
                "Return a full data profile for a dataset: row count, column roles, "
                "missing %, duplicate rows, and numeric statistics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "ID from list_datasets."}
                },
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": (
                "Generate a chart for EDA. You will receive the chart image so you can "
                "observe and describe what you see. The chart is also shown to the user. "
                "chart_type options: "
                "'histogram' — distribution of a numeric column (params: column, bins=30); "
                "'bar' — value counts of a categorical column (params: column, top_n=20); "
                "'scatter' — two numeric columns (params: x_column, y_column, color_column?); "
                "'correlation_heatmap' — correlation matrix of all numeric columns (no params needed); "
                "'box' — box plot of a numeric column, optionally grouped (params: column, group_by?); "
                "'missing_heatmap' — pattern of missing values across columns (no params needed)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "chart_type": {
                        "type": "string",
                        "enum": [
                            "histogram",
                            "bar",
                            "scatter",
                            "correlation_heatmap",
                            "box",
                            "missing_heatmap",
                        ],
                    },
                    "params": {
                        "type": "object",
                        "description": (
                            "Chart-specific parameters. "
                            "histogram: {column: str, bins: int}. "
                            "bar: {column: str, top_n: int}. "
                            "scatter: {x_column: str, y_column: str, color_column: str?}. "
                            "box: {column: str, group_by: str?}. "
                            "correlation_heatmap / missing_heatmap: {}."
                        ),
                    },
                },
                "required": ["dataset_id", "chart_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_derived_dataset",
            "description": (
                "Create a new dataset derived from an existing one. "
                "Operations: "
                "'drop_high_missing' — remove columns whose missing fraction exceeds params.threshold (default 0.5); "
                "'drop_duplicates' — remove fully duplicate rows; "
                "'select_columns' — keep only params.columns list; "
                "'filter_outliers' — IQR-based outlier row removal on all numeric columns; "
                "'encode_dates' — expand datetime columns into year/month/day/dayofweek features."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_dataset_id": {"type": "string"},
                    "new_name": {
                        "type": "string",
                        "description": "Human-readable name for the new dataset.",
                    },
                    "operation": {
                        "type": "string",
                        "enum": [
                            "drop_high_missing",
                            "drop_duplicates",
                            "select_columns",
                            "filter_outliers",
                            "encode_dates",
                        ],
                    },
                    "params": {
                        "type": "object",
                        "description": (
                            "Optional. drop_high_missing: {threshold: 0.5}. "
                            "select_columns: {columns: [str]}."
                        ),
                    },
                },
                "required": ["source_dataset_id", "new_name", "operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_model",
            "description": "Run AutoML training on a dataset targeting a specific column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "target_column": {"type": "string"},
                    "test_size": {"type": "number", "description": "Test fraction, default 0.2."},
                    "random_state": {"type": "integer", "description": "Random seed, default 42."},
                },
                "required": ["dataset_id", "target_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_strategy",
            "description": "Submit the final strategy report in markdown and end the analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Full markdown strategy report.",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]


class AiAutopilot:
    """Autonomously analyses all project datasets and runs AutoML via OpenAI function calling."""

    def __init__(self, api_key: str, project_id: str, store: ProjectStore) -> None:
        from openai import OpenAI  # deferred so import error is user-visible

        self._client = OpenAI(api_key=api_key)
        self._project_id = project_id
        self._store = store
        self._step_index = 0
        self.new_datasets: list[DatasetInfo] = []
        self.training_runs: list[dict[str, Any]] = []
        self.strategy_summary: str = ""

    def run(self) -> Iterator[AutopilotStep]:
        """Yield AutopilotStep objects as the AI works through the analysis."""
        datasets = self._store.list_datasets(self._project_id)
        project = self._store.get_project(self._project_id)

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Project: **{project.name}**\n"
                    f"There are {len(datasets)} dataset(s) available.\n\n"
                    "Please profile and chart every dataset, create improved datasets where "
                    "helpful, train the best models, and provide a final strategy report."
                ),
            },
        ]

        for _ in range(60):  # safety cap
            response = self._client.chat.completions.create(
                model="gpt-5.4",
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
            )
            choice = response.choices[0]

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": choice.message.content,
            }
            if choice.message.tool_calls:
                assistant_msg["tool_calls"] = [
                    tc.model_dump() for tc in choice.message.tool_calls
                ]
            messages.append(assistant_msg)

            if choice.message.content:
                yield self._make_step("thought", "AI Reasoning", choice.message.content)

            if choice.finish_reason == "stop":
                break

            if choice.finish_reason == "tool_calls":
                for tc in choice.message.tool_calls:
                    name = tc.function.name
                    args: dict[str, Any] = json.loads(tc.function.arguments or "{}")

                    yield self._make_step(
                        "tool_call", f"Calling: {name}", json.dumps(args, indent=2)
                    )

                    # _dispatch returns (tool_message_content, ui_step)
                    # tool_message_content is a str or list[dict] (vision)
                    tool_content, extra_step = self._dispatch(name, args)
                    if extra_step:
                        yield extra_step

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_content,
                        }
                    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_step(
        self,
        kind: str,
        title: str,
        detail: str = "",
        data: dict | None = None,
    ) -> AutopilotStep:
        self._step_index += 1
        return AutopilotStep(self._step_index, kind, title, detail, data)

    def _dispatch(
        self, name: str, args: dict[str, Any]
    ) -> tuple[str | list, AutopilotStep | None]:
        """Return (tool_message_content, ui_step).

        tool_message_content is a JSON string for normal tools, or a
        multimodal list[dict] for create_chart (text + image for vision).
        """
        if name == "list_datasets":
            return json.dumps(_to_json_safe(self._list_datasets())), None
        if name == "profile_dataset":
            result, step = self._profile_dataset(args.get("dataset_id", ""))
            return json.dumps(_to_json_safe(result)), step
        if name == "create_chart":
            return self._create_chart(args)
        if name == "create_derived_dataset":
            result, step = self._create_derived_dataset(args)
            return json.dumps(_to_json_safe(result)), step
        if name == "train_model":
            result, step = self._train_model(args)
            return json.dumps(_to_json_safe(result)), step
        if name == "finalize_strategy":
            self.strategy_summary = args.get("summary", "")
            return (
                json.dumps({"status": "done"}),
                self._make_step("summary", "Strategy Finalised", self.strategy_summary),
            )
        return json.dumps({"error": f"Unknown tool: {name}"}), None

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _list_datasets(self) -> dict[str, Any]:
        datasets = self._store.list_datasets(self._project_id)
        return {
            "datasets": [
                {
                    "id": ds.id,
                    "name": ds.name,
                    "rows": ds.row_count,
                    "columns": ds.column_count,
                    "type": ds.source_type,
                }
                for ds in datasets
            ]
        }

    def _profile_dataset(
        self, dataset_id: str
    ) -> tuple[dict[str, Any], AutopilotStep | None]:
        ds = self._find_dataset(dataset_id)
        if ds is None:
            return {"error": f"Dataset '{dataset_id}' not found."}, None

        loaded = load_dataset(ds.file_path, ds.table_name)
        profile = profile_dataframe(loaded.dataframe)

        trimmed_columns = [
            {**col, "sample_values": col.get("sample_values", [])[:3]}
            for col in profile.get("columns", [])
        ]
        trimmed_profile = {**profile, "columns": trimmed_columns}

        step = self._make_step(
            "tool_result",
            f"Profiled: {ds.name}",
            (
                f"{profile['row_count']} rows × {profile['column_count']} cols | "
                f"{profile['missing_pct']:.1f}% missing | "
                f"{profile['duplicate_rows']} duplicates"
            ),
            data={"dataset_name": ds.name, "profile": _to_json_safe(trimmed_profile)},
        )
        return _to_json_safe(trimmed_profile), step

    def _create_chart(
        self, args: dict[str, Any]
    ) -> tuple[str | list, AutopilotStep | None]:
        """Build the requested chart and return a multimodal tool response with the PNG image."""
        dataset_id: str = args.get("dataset_id", "")
        chart_type: str = args.get("chart_type", "")
        params: dict = args.get("params") or {}

        ds = self._find_dataset(dataset_id)
        if ds is None:
            return json.dumps({"error": f"Dataset '{dataset_id}' not found."}), None

        loaded = load_dataset(ds.file_path, ds.table_name)
        df = loaded.dataframe

        fig, title, description = _build_figure(df, ds.name, chart_type, params)
        if fig is None:
            return json.dumps({"error": description}), None

        step = self._make_step(
            "chart",
            title,
            description,
            data={"figure": fig, "dataset_name": ds.name},
        )

        result_text = json.dumps(
            {"chart": title, "description": description, "dataset": ds.name}
        )

        # Try to render PNG for vision feedback to the model
        b64_img = _fig_to_base64(fig)
        if b64_img:
            tool_content: str | list = [
                {"type": "text", "text": result_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                },
            ]
        else:
            tool_content = result_text

        return tool_content, step

    def _create_derived_dataset(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], AutopilotStep | None]:
        source_id: str = args.get("source_dataset_id", "")
        new_name: str = args.get("new_name", "derived")
        operation: str = args.get("operation", "")
        params: dict = args.get("params") or {}

        source = self._find_dataset(source_id)
        if source is None:
            return {"error": f"Dataset '{source_id}' not found."}, None

        loaded = load_dataset(source.file_path, source.table_name)
        df = loaded.dataframe.copy()
        detail = ""

        if operation == "drop_high_missing":
            threshold = float(params.get("threshold", 0.5))
            before = df.shape[1]
            df = df.loc[:, df.isnull().mean() <= threshold]
            detail = f"Dropped {before - df.shape[1]} columns with >{threshold * 100:.0f}% missing"

        elif operation == "drop_duplicates":
            before = len(df)
            df = df.drop_duplicates()
            detail = f"Removed {before - len(df)} duplicate rows"

        elif operation == "select_columns":
            cols = [c for c in params.get("columns", []) if c in df.columns]
            if not cols:
                return {"error": "None of the specified columns exist in the dataset."}, None
            df = df[cols]
            detail = f"Selected {len(cols)} columns"

        elif operation == "filter_outliers":
            numeric_cols = df.select_dtypes(include="number").columns
            before = len(df)
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    df = df[
                        (df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)
                    ]
            detail = f"Removed {before - len(df)} outlier rows (IQR method)"

        elif operation == "encode_dates":
            dt_cols = df.select_dtypes(
                include=["datetime64[ns]", "datetime64"]
            ).columns.tolist()
            for col in dt_cols:
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df = df.drop(columns=dt_cols)
            detail = f"Expanded {len(dt_cols)} datetime column(s) → year/month/day/dayofweek"

        else:
            return {"error": f"Unknown operation: {operation}"}, None

        if df.empty:
            return {"error": "Derived dataset is empty after applying the operation."}, None

        csv_bytes = df.to_csv(index=False).encode()
        filename = f"{new_name.replace(' ', '_')}.csv"
        saved_path = self._store.save_dataset_file(self._project_id, filename, csv_bytes)
        ds_info = self._store.register_dataset(
            self._project_id,
            name=new_name,
            source_name=filename,
            source_type="csv",
            file_path=str(saved_path),
            row_count=int(len(df)),
            column_count=int(len(df.columns)),
        )
        self.new_datasets.append(ds_info)

        step = self._make_step(
            "new_dataset",
            f"New Dataset: {new_name}",
            f"{detail} → {len(df)} rows × {len(df.columns)} cols",
            data={
                "dataset_id": ds_info.id,
                "rows": int(len(df)),
                "cols": int(len(df.columns)),
                "operation": operation,
            },
        )
        return {
            "dataset_id": ds_info.id,
            "name": new_name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "detail": detail,
        }, step

    def _train_model(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], AutopilotStep | None]:
        dataset_id: str = args.get("dataset_id", "")
        target_column: str = args.get("target_column", "")
        test_size: float = float(args.get("test_size", 0.2))
        random_state: int = int(args.get("random_state", 42))

        ds = self._find_dataset(dataset_id)
        if ds is None:
            return {"error": f"Dataset '{dataset_id}' not found."}, None

        loaded = load_dataset(ds.file_path, ds.table_name)
        settings = TrainingSettings(
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
        )

        result, model = train_automl(loaded.dataframe, settings)
        project = self._store.get_project(self._project_id)
        profile = profile_dataframe(loaded.dataframe)

        metadata = result.to_metadata()
        metadata["dataset"] = ds.to_dict()
        report_text = build_markdown_report(project.name, ds.to_dict(), metadata, profile)
        self._store.save_run(self._project_id, metadata, model, report_text)

        run_summary = {
            "run_id": result.run_id,
            "dataset": ds.name,
            "target": target_column,
            "task_type": result.task_type,
            "best_model": result.best_model_name,
            "best_metrics": result.best_metrics,
        }
        self.training_runs.append(run_summary)

        metrics_str = ", ".join(
            f"{k}: {v:.4f}" for k, v in result.best_metrics.items()
        )
        step = self._make_step(
            "training",
            f"Trained: {ds.name} → {target_column}",
            (
                f"Task: {result.task_type} | "
                f"Best model: {result.best_model_name} | "
                f"{metrics_str}"
            ),
            data=_to_json_safe(run_summary),
        )
        return (
            _to_json_safe(
                {
                    "run_id": result.run_id,
                    "task_type": result.task_type,
                    "best_model": result.best_model_name,
                    "best_metrics": result.best_metrics,
                }
            ),
            step,
        )

    def _find_dataset(self, dataset_id: str) -> DatasetInfo | None:
        return next(
            (d for d in self._store.list_datasets(self._project_id) if d.id == dataset_id),
            None,
        )


# ------------------------------------------------------------------
# Chart building
# ------------------------------------------------------------------


def _build_figure(
    df: pd.DataFrame, dataset_name: str, chart_type: str, params: dict
) -> tuple[go.Figure | None, str, str]:
    """Return (figure, title, description). figure is None on error, description is the error msg."""
    try:
        if chart_type == "histogram":
            col = params.get("column")
            if col not in df.columns:
                return None, "", f"Column '{col}' not found."
            bins = int(params.get("bins", 30))
            fig = px.histogram(
                df, x=col, nbins=bins,
                title=f"Distribution of {col}",
                template="plotly_white",
            )
            return fig, f"Histogram: {col}", f"Distribution of {col} in {dataset_name}"

        if chart_type == "bar":
            col = params.get("column")
            if col not in df.columns:
                return None, "", f"Column '{col}' not found."
            top_n = int(params.get("top_n", 20))
            counts = df[col].value_counts().head(top_n).reset_index()
            counts.columns = [col, "count"]
            fig = px.bar(
                counts, x=col, y="count",
                title=f"Value Counts: {col} (top {top_n})",
                template="plotly_white",
            )
            return fig, f"Bar: {col}", f"Top {top_n} value counts of {col} in {dataset_name}"

        if chart_type == "scatter":
            x_col = params.get("x_column")
            y_col = params.get("y_column")
            if x_col not in df.columns or y_col not in df.columns:
                return None, "", "x_column or y_column not found."
            color_col = params.get("color_column")
            sample = df.head(2000)
            fig = px.scatter(
                sample,
                x=x_col, y=y_col,
                color=color_col if color_col and color_col in df.columns else None,
                title=f"Scatter: {x_col} vs {y_col}",
                template="plotly_white",
                opacity=0.6,
            )
            return (
                fig,
                f"Scatter: {x_col} vs {y_col}",
                f"Scatter of {x_col} vs {y_col} in {dataset_name}",
            )

        if chart_type == "correlation_heatmap":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.shape[1] < 2:
                return None, "", "Need at least 2 numeric columns for a correlation heatmap."
            corr = numeric_df.corr()
            fig = px.imshow(
                corr,
                title=f"Correlation Matrix — {dataset_name}",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                text_auto=".2f",
                template="plotly_white",
            )
            return (
                fig,
                "Correlation Heatmap",
                f"Correlation of {numeric_df.shape[1]} numeric columns in {dataset_name}",
            )

        if chart_type == "box":
            col = params.get("column")
            if col not in df.columns:
                return None, "", f"Column '{col}' not found."
            group_by = params.get("group_by")
            x_arg = group_by if group_by and group_by in df.columns else None
            fig = px.box(
                df, y=col, x=x_arg,
                title=f"Box Plot: {col}",
                template="plotly_white",
            )
            label = f"Box: {col}" + (f" by {group_by}" if x_arg else "")
            return fig, label, f"Box plot of {col} in {dataset_name}"

        if chart_type == "missing_heatmap":
            missing_mask = df.isnull()
            cols_with_missing = missing_mask.columns[missing_mask.any()].tolist()
            if not cols_with_missing:
                return None, "", "No missing values found — nothing to chart."
            sample = missing_mask[cols_with_missing].head(100).astype(int)
            fig = px.imshow(
                sample.T,
                title=f"Missing Value Pattern — {dataset_name} (first 100 rows)",
                color_continuous_scale=[[0, "#f0f0f0"], [1, "#e53e3e"]],
                labels={"x": "Row", "y": "Column", "color": "Missing"},
                template="plotly_white",
            )
            return (
                fig,
                "Missing Data Heatmap",
                f"Pattern of missing values in {len(cols_with_missing)} columns of {dataset_name}",
            )

        return None, "", f"Unknown chart_type: {chart_type}"

    except Exception as exc:
        return None, "", f"Chart error: {exc}"


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------


def _fig_to_base64(fig: go.Figure) -> str | None:
    """Render a Plotly figure to a base64 PNG for vision feedback. Returns None on failure."""
    try:
        import plotly.io as pio

        img_bytes = pio.to_image(fig, format="png", width=900, height=480, scale=1)
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        return None


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serialisable Python natives."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (v != v) else v  # NaN → None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _to_json_safe(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj
