"""AI autopilot: uses OpenAI function calling to autonomously analyse data and run AutoML."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Generator

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
    # "thought"|"tool_call"|"tool_result"|"chart"|"ask"|"new_dataset"|"training"|"summary"
    kind: str
    title: str
    detail: str = ""
    data: dict[str, Any] | None = None


_SYSTEM_PROMPT = """\
You are an expert ML analyst with full autonomous control over an AutoML discovery platform.
You have vision capability — when you create a chart you receive the image and can describe
what you observe to inform your strategy.

YOUR MISSION — follow this order precisely:

STEP 1 — UNDERSTAND GOALS
  Call list_datasets to see what is available.
  Then call ask_user with 3–5 focused questions covering:
    • What business outcome or prediction does the user want?
    • Are there columns to exclude or a specific target they have in mind?
    • Any domain knowledge about the data (e.g. known relationships, data quality issues)?
    • Accuracy vs. interpretability preference?
  Wait for the user's answers before proceeding.

STEP 2 — EXPLORE THE DATA
  For every dataset: call profile_dataset, then call create_chart to visualise it:
    - correlation_heatmap (2+ numeric columns)
    - histogram for each candidate numeric target
    - bar for categorical columns with ≤20 unique values
    - missing_heatmap if any missing values exist
    - scatter between pairs of strongly correlated numeric columns
  After each chart, describe what you observe and what it means for modelling.

STEP 3 — PROPOSE MULTIPLE APPROACHES
  Based on the profile, charts, and user goals, reason explicitly about 2–3 distinct
  modelling strategies (e.g. different targets, classification vs regression, different
  feature sets). Describe the tradeoff of each. Conclude with which you recommend and why.

STEP 4 — PREPARE DATA
  Create derived/cleaned datasets where beneficial (drop high-missing cols, remove
  outliers, encode dates). Always explain your reasoning.

STEP 5 — TRAIN
  Execute train_model for the best dataset+target pair(s) based on your analysis.
  If you identified multiple viable strategies, train the top two.

STEP 6 — SUMMARISE
  Call finalize_strategy with a comprehensive markdown report that covers:
  EDA observations (cite specific chart findings), derived datasets created and why,
  the strategies you considered, training results, and specific actionable recommendations.
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
            "name": "ask_user",
            "description": (
                "Pause and ask the user clarifying questions about their goals, domain knowledge, "
                "and constraints. Call this ONCE early in the analysis, before EDA. "
                "The user will answer each question in order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of 3–5 specific questions for the user.",
                    }
                },
                "required": ["questions"],
            },
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
                "'correlation_heatmap' — correlation matrix of all numeric columns; "
                "'box' — box plot of a numeric column (params: column, group_by?); "
                "'missing_heatmap' — pattern of missing values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "chart_type": {
                        "type": "string",
                        "enum": [
                            "histogram", "bar", "scatter",
                            "correlation_heatmap", "box", "missing_heatmap",
                        ],
                    },
                    "params": {
                        "type": "object",
                        "description": (
                            "histogram: {column, bins?}. bar: {column, top_n?}. "
                            "scatter: {x_column, y_column, color_column?}. "
                            "box: {column, group_by?}. heatmaps: {}."
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
                "'drop_high_missing' — remove columns > params.threshold missing (default 0.5); "
                "'drop_duplicates' — remove fully duplicate rows; "
                "'select_columns' — keep only params.columns list; "
                "'filter_outliers' — IQR-based outlier row removal on numeric columns; "
                "'encode_dates' — expand datetime cols to year/month/day/dayofweek."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_dataset_id": {"type": "string"},
                    "new_name": {"type": "string"},
                    "operation": {
                        "type": "string",
                        "enum": [
                            "drop_high_missing", "drop_duplicates",
                            "select_columns", "filter_outliers", "encode_dates",
                        ],
                    },
                    "params": {"type": "object"},
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
                    "test_size": {"type": "number", "description": "Default 0.2."},
                    "random_state": {"type": "integer", "description": "Default 42."},
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
                    "summary": {"type": "string", "description": "Full markdown strategy report."}
                },
                "required": ["summary"],
            },
        },
    },
]


class AiAutopilot:
    """Autonomously analyses all project datasets and runs AutoML via OpenAI function calling."""

    def __init__(
        self,
        api_key: str,
        project_id: str,
        store: ProjectStore,
        user_goal: str = "",
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._project_id = project_id
        self._store = store
        self._user_goal = user_goal
        self._step_index = 0
        self.new_datasets: list[DatasetInfo] = []
        self.training_runs: list[dict[str, Any]] = []
        self.strategy_summary: str = ""

    def run(self) -> Generator[AutopilotStep, list[str] | None, None]:
        """Yield AutopilotStep objects. On an 'ask' step, send() the list of answer strings."""
        datasets = self._store.list_datasets(self._project_id)
        project = self._store.get_project(self._project_id)

        goal_line = f"\nUser's stated goal: {self._user_goal}" if self._user_goal else ""
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Project: **{project.name}**\n"
                    f"Datasets available: {len(datasets)}.{goal_line}\n\n"
                    "Please begin by listing datasets and asking me your questions."
                ),
            },
        ]

        for _ in range(80):  # safety cap
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

                    # ── ask_user: pause and wait for answers via send() ──────────
                    if name == "ask_user":
                        questions: list[str] = args.get("questions", [])
                        ask_step = self._make_step(
                            "ask",
                            "AI has questions for you",
                            f"{len(questions)} question(s)",
                            data={"questions": questions},
                        )
                        answers: list[str] | None = yield ask_step
                        if not answers:
                            answers = [""] * len(questions)
                        answer_text = "\n".join(
                            f"Q{i+1}: {q}\nA{i+1}: {a}"
                            for i, (q, a) in enumerate(zip(questions, answers))
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps({"user_answers": answer_text}),
                            }
                        )
                        break  # process remaining outer loop iteration fresh

                    # ── all other tools ──────────────────────────────────────────
                    tool_content, extra_step = self._dispatch(name, args)
                    if extra_step:
                        yield extra_step
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": tool_content}
                    )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self, name: str, args: dict[str, Any]
    ) -> tuple[str | list, AutopilotStep | None]:
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
        return {
            "datasets": [
                {"id": ds.id, "name": ds.name, "rows": ds.row_count,
                 "columns": ds.column_count, "type": ds.source_type}
                for ds in self._store.list_datasets(self._project_id)
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
        trimmed = {
            **profile,
            "columns": [
                {**col, "sample_values": col.get("sample_values", [])[:3]}
                for col in profile.get("columns", [])
            ],
        }
        step = self._make_step(
            "tool_result",
            f"Profiled: {ds.name}",
            (
                f"{profile['row_count']} rows × {profile['column_count']} cols | "
                f"{profile['missing_pct']:.1f}% missing | "
                f"{profile['duplicate_rows']} duplicates"
            ),
            data={"dataset_name": ds.name, "profile": _to_json_safe(trimmed)},
        )
        return _to_json_safe(trimmed), step

    def _create_chart(
        self, args: dict[str, Any]
    ) -> tuple[str | list, AutopilotStep | None]:
        ds = self._find_dataset(args.get("dataset_id", ""))
        if ds is None:
            return json.dumps({"error": f"Dataset '{args.get('dataset_id')}' not found."}), None
        loaded = load_dataset(ds.file_path, ds.table_name)
        fig, title, description = _build_figure(
            loaded.dataframe, ds.name,
            args.get("chart_type", ""),
            args.get("params") or {},
        )
        if fig is None:
            return json.dumps({"error": description}), None

        step = self._make_step(
            "chart", title, description,
            data={"figure": fig, "dataset_name": ds.name},
        )
        result_text = json.dumps(
            {"chart": title, "description": description, "dataset": ds.name}
        )
        b64 = _fig_to_base64(fig)
        if b64:
            tool_content: str | list = [
                {"type": "text", "text": result_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        else:
            tool_content = result_text
        return tool_content, step

    def _create_derived_dataset(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], AutopilotStep | None]:
        source = self._find_dataset(args.get("source_dataset_id", ""))
        if source is None:
            return {"error": f"Dataset '{args.get('source_dataset_id')}' not found."}, None
        loaded = load_dataset(source.file_path, source.table_name)
        df = loaded.dataframe.copy()
        operation: str = args.get("operation", "")
        params: dict = args.get("params") or {}
        new_name: str = args.get("new_name", "derived")
        detail = ""

        if operation == "drop_high_missing":
            threshold = float(params.get("threshold", 0.5))
            before = df.shape[1]
            df = df.loc[:, df.isnull().mean() <= threshold]
            detail = f"Dropped {before - df.shape[1]} cols with >{threshold*100:.0f}% missing"
        elif operation == "drop_duplicates":
            before = len(df)
            df = df.drop_duplicates()
            detail = f"Removed {before - len(df)} duplicate rows"
        elif operation == "select_columns":
            cols = [c for c in params.get("columns", []) if c in df.columns]
            if not cols:
                return {"error": "None of the specified columns exist."}, None
            df = df[cols]
            detail = f"Selected {len(cols)} columns"
        elif operation == "filter_outliers":
            before = len(df)
            for col in df.select_dtypes(include="number").columns:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
            detail = f"Removed {before - len(df)} outlier rows (IQR)"
        elif operation == "encode_dates":
            dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
            for col in dt_cols:
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df = df.drop(columns=dt_cols)
            detail = f"Expanded {len(dt_cols)} datetime col(s) → year/month/day/dayofweek"
        else:
            return {"error": f"Unknown operation: {operation}"}, None

        if df.empty:
            return {"error": "Derived dataset is empty."}, None

        csv_bytes = df.to_csv(index=False).encode()
        filename = f"{new_name.replace(' ', '_')}.csv"
        saved = self._store.save_dataset_file(self._project_id, filename, csv_bytes)
        ds_info = self._store.register_dataset(
            self._project_id, name=new_name, source_name=filename,
            source_type="csv", file_path=str(saved),
            row_count=int(len(df)), column_count=int(len(df.columns)),
        )
        self.new_datasets.append(ds_info)
        step = self._make_step(
            "new_dataset", f"New Dataset: {new_name}",
            f"{detail} → {len(df)} rows × {len(df.columns)} cols",
            data={"dataset_id": ds_info.id, "rows": int(len(df)), "cols": int(len(df.columns))},
        )
        return {
            "dataset_id": ds_info.id, "name": new_name,
            "rows": int(len(df)), "columns": int(len(df.columns)), "detail": detail,
        }, step

    def _train_model(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], AutopilotStep | None]:
        ds = self._find_dataset(args.get("dataset_id", ""))
        if ds is None:
            return {"error": f"Dataset '{args.get('dataset_id')}' not found."}, None
        loaded = load_dataset(ds.file_path, ds.table_name)
        target = args.get("target_column", "")
        settings = TrainingSettings(
            target_column=target,
            test_size=float(args.get("test_size", 0.2)),
            random_state=int(args.get("random_state", 42)),
        )
        result, model = train_automl(loaded.dataframe, settings)
        project = self._store.get_project(self._project_id)
        profile = profile_dataframe(loaded.dataframe)
        metadata = result.to_metadata()
        metadata["dataset"] = ds.to_dict()
        report_text = build_markdown_report(project.name, ds.to_dict(), metadata, profile)
        self._store.save_run(self._project_id, metadata, model, report_text)

        summary = {
            "run_id": result.run_id, "dataset": ds.name, "target": target,
            "task_type": result.task_type, "best_model": result.best_model_name,
            "best_metrics": result.best_metrics,
        }
        self.training_runs.append(summary)
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in result.best_metrics.items())
        step = self._make_step(
            "training", f"Trained: {ds.name} → {target}",
            f"Task: {result.task_type} | Best: {result.best_model_name} | {metrics_str}",
            data=_to_json_safe(summary),
        )
        return _to_json_safe({
            "run_id": result.run_id, "task_type": result.task_type,
            "best_model": result.best_model_name, "best_metrics": result.best_metrics,
        }), step

    def _find_dataset(self, dataset_id: str) -> DatasetInfo | None:
        return next(
            (d for d in self._store.list_datasets(self._project_id) if d.id == dataset_id),
            None,
        )

    def _make_step(
        self, kind: str, title: str, detail: str = "", data: dict | None = None
    ) -> AutopilotStep:
        self._step_index += 1
        return AutopilotStep(self._step_index, kind, title, detail, data)


# ------------------------------------------------------------------
# Chart building
# ------------------------------------------------------------------


def _build_figure(
    df: pd.DataFrame, dataset_name: str, chart_type: str, params: dict
) -> tuple[go.Figure | None, str, str]:
    try:
        if chart_type == "histogram":
            col = params.get("column")
            if col not in df.columns:
                return None, "", f"Column '{col}' not found."
            fig = px.histogram(df, x=col, nbins=int(params.get("bins", 30)),
                               title=f"Distribution of {col}", template="plotly_white")
            return fig, f"Histogram: {col}", f"Distribution of {col} in {dataset_name}"

        if chart_type == "bar":
            col = params.get("column")
            if col not in df.columns:
                return None, "", f"Column '{col}' not found."
            top_n = int(params.get("top_n", 20))
            counts = df[col].value_counts().head(top_n).reset_index()
            counts.columns = [col, "count"]
            fig = px.bar(counts, x=col, y="count",
                         title=f"Value Counts: {col} (top {top_n})", template="plotly_white")
            return fig, f"Bar: {col}", f"Top {top_n} value counts of {col} in {dataset_name}"

        if chart_type == "scatter":
            x_col, y_col = params.get("x_column"), params.get("y_column")
            if x_col not in df.columns or y_col not in df.columns:
                return None, "", "x_column or y_column not found."
            color_col = params.get("color_column")
            fig = px.scatter(
                df.head(2000), x=x_col, y=y_col,
                color=color_col if color_col and color_col in df.columns else None,
                title=f"Scatter: {x_col} vs {y_col}", template="plotly_white", opacity=0.6,
            )
            return fig, f"Scatter: {x_col} vs {y_col}", f"Scatter of {x_col} vs {y_col} in {dataset_name}"

        if chart_type == "correlation_heatmap":
            num = df.select_dtypes(include="number")
            if num.shape[1] < 2:
                return None, "", "Need ≥2 numeric columns."
            corr = num.corr()
            fig = px.imshow(corr, title=f"Correlation — {dataset_name}",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            text_auto=".2f", template="plotly_white")
            return fig, "Correlation Heatmap", f"Correlation of {num.shape[1]} numeric cols in {dataset_name}"

        if chart_type == "box":
            col = params.get("column")
            if col not in df.columns:
                return None, "", f"Column '{col}' not found."
            gb = params.get("group_by")
            fig = px.box(df, y=col, x=gb if gb and gb in df.columns else None,
                         title=f"Box Plot: {col}", template="plotly_white")
            label = f"Box: {col}" + (f" by {gb}" if gb and gb in df.columns else "")
            return fig, label, f"Box plot of {col} in {dataset_name}"

        if chart_type == "missing_heatmap":
            mask = df.isnull()
            missing_cols = mask.columns[mask.any()].tolist()
            if not missing_cols:
                return None, "", "No missing values — nothing to chart."
            sample = mask[missing_cols].head(100).astype(int)
            fig = px.imshow(sample.T, title=f"Missing Values — {dataset_name} (first 100 rows)",
                            color_continuous_scale=[[0, "#f0f0f0"], [1, "#e53e3e"]],
                            labels={"x": "Row", "y": "Column", "color": "Missing"},
                            template="plotly_white")
            return fig, "Missing Heatmap", f"Missing pattern in {len(missing_cols)} cols of {dataset_name}"

        return None, "", f"Unknown chart_type: {chart_type}"
    except Exception as exc:
        return None, "", f"Chart error: {exc}"


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------


def _fig_to_base64(fig: go.Figure) -> str | None:
    """Render figure to base64 PNG for vision feedback. Returns None on failure."""
    try:
        import plotly.io as pio
        img_bytes = pio.to_image(fig, format="png", width=900, height=480, scale=1)
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        return None


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serialisable natives."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (v != v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _to_json_safe(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj
