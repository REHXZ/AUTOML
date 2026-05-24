"""EDA Agent: profiles datasets and generates charts with vision feedback."""

from __future__ import annotations

import json
import logging
from typing import Any, Generator

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..ingestion import load_dataset
from ..logging_setup import configure_logging
from ..profiling import profile_dataframe
from .base import (
    AgentContext,
    AutopilotStep,
    BaseAgent,
    to_json_safe,
    vision_tool_content,
)

configure_logging()
log = logging.getLogger(__name__)

# Chart types that require at least one column parameter in params.
_PARAM_REQUIRED_CHARTS = frozenset({"histogram", "bar", "scatter", "box", "violin", "pairplot"})
# All known per-chart param keys — used to detect flat args from LLM.
_KNOWN_PARAM_KEYS = frozenset({"column", "bins", "top_n", "x_column", "y_column",
                                "color_column", "group_by", "columns"})


_SYSTEM_PROMPT = """\
You are the EDA Agent — an expert exploratory data analyst working under the
direction of the AIML Scientist.

Your job is to deeply UNDERSTAND a dataset and report back useful, specific
observations the Scientist can act on.

How to work:
1. Call profile_dataset to understand shape, types, missingness, and stats.
2. Call create_chart REPEATEDLY to visualise distributions, correlations,
   missing patterns, candidate-target relationships, and anything else that
   looks interesting. You have vision — describe what you SEE in each image:
   skew, outliers, clusters, class imbalance, leakage hints, multimodality.
3. After every chart, write a one-paragraph observation: what does this tell
   us about modelling strategy?
4. When you have a coherent picture, call record_finding(text) to leave a
   short bullet-point note in the shared notebook for the Scientist.
5. When done, call done(summary) with a structured JSON of your key
   observations: candidate targets, problematic columns, recommended
   transformations, suspected leakage, suggested next moves.

Be thorough. Do not stop after two charts — explore every angle the
Scientist's instructions imply.
"""


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "profile_dataset",
                "description": "Return row count, column roles, missing %, duplicates, and numeric stats.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
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
                    "Generate a chart and receive the rendered image for visual analysis. "
                    "chart_type: 'histogram' (params: column, bins?); "
                    "'bar' (params: column, top_n?); "
                    "'scatter' (params: x_column, y_column, color_column?); "
                    "'correlation_heatmap'; "
                    "'box' (params: column, group_by?); "
                    "'missing_heatmap'; "
                    "'violin' (params: column, group_by?); "
                    "'pairplot' (params: columns, color_column?) — up to 4 numeric columns."
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
                                "violin", "pairplot",
                            ],
                        },
                        "params": {"type": "object"},
                    },
                    "required": ["dataset_id", "chart_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_finding",
                "description": "Write a single observation bullet to the shared notebook.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "done",
                "description": "Report finished. Provide a structured JSON summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "candidate_targets": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "problematic_columns": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "recommended_transformations": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "suspected_leakage": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "next_moves": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "narrative": {"type": "string"},
                    },
                    "required": ["narrative"],
                },
            },
        },
    ]


class EdaAgent(BaseAgent):
    """Profiles datasets and creates EDA charts on the Scientist's behalf."""

    name = "eda"
    display_name = "EDA Agent"

    def __init__(self, client, deployment: str, context: AgentContext) -> None:
        super().__init__(client, deployment, context)
        self._summary: dict[str, Any] = {}

    def run(
        self, instructions: str
    ) -> Generator[AutopilotStep, list[str] | None, dict[str, Any]]:
        yield self._step(
            "agent_start",
            "EDA Agent dispatched",
            instructions or "(no specific instructions — explore broadly)",
        )

        datasets = self._ctx.list_datasets()
        dataset_index = "\n".join(
            f"- id={d.id} name={d.name} rows={d.row_count} cols={d.column_count}"
            for d in datasets
        ) or "(no datasets registered)"

        user_prompt = (
            f"Scientist's instructions:\n{instructions}\n\n"
            f"Available datasets:\n{dataset_index}\n\n"
            f"Project user-stated goal: {self._ctx.user_goal or '(none)'}\n\n"
            "Begin exploration."
        )

        yield from self.run_llm_loop(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=_tools(),
            dispatch=self._dispatch,
            max_iterations=30,
            thought_title="EDA Agent — Reasoning",
        )

        yield self._step("agent_end", "EDA Agent finished", "")
        return self._summary or {"narrative": "EDA agent ended without summary."}

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self, name: str, args: dict, tool_call_id: str
    ) -> tuple[str | list | None, AutopilotStep | None, bool]:
        if name == "profile_dataset":
            return self._profile(args.get("dataset_id", ""))
        if name == "create_chart":
            return self._chart(args)
        if name == "record_finding":
            return self._record(args.get("text", ""))
        if name == "done":
            self._summary = to_json_safe(args)
            return json.dumps({"status": "noted"}), None, True
        return json.dumps({"error": f"Unknown tool: {name}"}), None, False

    def _profile(self, dataset_id: str) -> tuple[str, AutopilotStep | None, bool]:
        ds = self._ctx.find_dataset(dataset_id)
        if ds is None:
            return json.dumps({"error": f"Dataset '{dataset_id}' not found."}), None, False
        loaded = load_dataset(ds.file_path, ds.table_name)
        profile = profile_dataframe(loaded.dataframe)
        trimmed = {
            **profile,
            "columns": [
                {**col, "sample_values": col.get("sample_values", [])[:3]}
                for col in profile.get("columns", [])
            ],
        }
        step = self._step(
            "tool_result",
            f"Profiled: {ds.name}",
            (
                f"{profile['row_count']} rows × {profile['column_count']} cols | "
                f"{profile['missing_pct']:.1f}% missing | "
                f"{profile['duplicate_rows']} duplicates"
            ),
            data={"dataset_name": ds.name, "profile": to_json_safe(trimmed)},
        )
        return json.dumps(to_json_safe(trimmed)), step, False

    def _chart(self, args: dict) -> tuple[str | list, AutopilotStep | None, bool]:
        dataset_id = args.get("dataset_id", "")
        chart_type = args.get("chart_type", "")
        params: dict = args.get("params") or {}

        # ── Flat-params fallback ──────────────────────────────────────────────
        # The LLM sometimes sends chart-specific keys at the top level of the
        # tool call instead of nested under "params".  Detect and promote them
        # so per-column charts still work, and emit a warning so we can track it.
        if not params:
            flat = {k: v for k, v in args.items() if k in _KNOWN_PARAM_KEYS}
            if flat:
                log.warning(
                    "create_chart | flat-params detected for chart_type=%r — "
                    "LLM sent %s at top level instead of nested under 'params'; promoting.",
                    chart_type, list(flat),
                )
                params = flat

        if not params and chart_type in _PARAM_REQUIRED_CHARTS:
            log.warning(
                "create_chart | chart_type=%r requires params but received none — "
                "full args: %s",
                chart_type, args,
            )

        ds = self._ctx.find_dataset(dataset_id)
        if ds is None:
            log.warning("create_chart | dataset_id=%r not found", dataset_id)
            return (
                json.dumps({"error": f"Dataset '{dataset_id}' not found."}),
                None,
                False,
            )

        log.info(
            "create_chart | dataset=%s chart_type=%r params=%s",
            ds.name, chart_type, params,
        )

        loaded = load_dataset(ds.file_path, ds.table_name)
        fig, title, description = _build_figure(
            loaded.dataframe,
            ds.name,
            chart_type,
            params,
        )
        if fig is None:
            log.warning(
                "create_chart | FAILED dataset=%s chart_type=%r error=%r",
                ds.name, chart_type, description,
            )
            return json.dumps({"error": description}), None, False

        log.info("create_chart | OK dataset=%s chart_type=%r title=%r", ds.name, chart_type, title)
        step = self._step(
            "chart",
            title,
            description,
            data={
                "figure": fig,
                "dataset_name": ds.name,
                "dataset_id": ds.id,
                "chart_type": chart_type,
                "chart_params": dict(params),
            },
        )
        result_text = json.dumps(
            {"chart": title, "description": description, "dataset": ds.name}
        )
        return vision_tool_content(result_text, fig), step, False

    def _record(self, text: str) -> tuple[str, AutopilotStep, bool]:
        text = (text or "").strip()
        if text:
            self._ctx.notebook.append(f"[EDA] {text}")
        return (
            json.dumps({"recorded": True, "notebook_size": len(self._ctx.notebook)}),
            self._step("observation", "EDA finding", text),
            False,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Chart construction
# ──────────────────────────────────────────────────────────────────────────────


def _build_figure(
    df: pd.DataFrame, dataset_name: str, chart_type: str, params: dict
) -> tuple[go.Figure | None, str, str]:
    log.debug(
        "_build_figure | dataset=%s chart_type=%r params=%s df_cols=%s",
        dataset_name, chart_type, params, list(df.columns)[:20],
    )
    try:
        if chart_type == "histogram":
            col = params.get("column")
            if not col or col not in df.columns:
                log.warning(
                    "_build_figure | histogram: column=%r not in dataset=%s (available: %s)",
                    col, dataset_name, list(df.columns)[:10],
                )
                return None, "", f"Column '{col}' not found."
            fig = px.histogram(
                df, x=col, nbins=int(params.get("bins", 30)),
                title=f"Distribution of {col}", template="plotly_white",
            )
            return fig, f"Histogram: {col}", f"Distribution of {col} in {dataset_name}"

        if chart_type == "bar":
            col = params.get("column")
            if not col or col not in df.columns:
                log.warning(
                    "_build_figure | bar: column=%r not in dataset=%s", col, dataset_name
                )
                return None, "", f"Column '{col}' not found."
            top_n = int(params.get("top_n", 20))
            vc = df[col].value_counts().head(top_n).reset_index()
            # Normalise column names across pandas versions:
            # pandas 1.x → ['index', col]; pandas 2.x → [col, 'count']
            vc.columns = ["value", "count"]
            fig = px.bar(
                vc, x="value", y="count",
                title=f"Value Counts: {col} (top {top_n})", template="plotly_white",
                labels={"value": col},
            )
            return fig, f"Bar: {col}", f"Top {top_n} value counts of {col} in {dataset_name}"

        if chart_type == "scatter":
            x_col = params.get("x_column")
            y_col = params.get("y_column")
            if not x_col or x_col not in df.columns:
                log.warning(
                    "_build_figure | scatter: x_column=%r not in dataset=%s", x_col, dataset_name
                )
                return None, "", f"x_column '{x_col}' not found."
            if not y_col or y_col not in df.columns:
                log.warning(
                    "_build_figure | scatter: y_column=%r not in dataset=%s", y_col, dataset_name
                )
                return None, "", f"y_column '{y_col}' not found."
            color_col = params.get("color_column")
            fig = px.scatter(
                df.head(2000), x=x_col, y=y_col,
                color=color_col if color_col and color_col in df.columns else None,
                title=f"Scatter: {x_col} vs {y_col}",
                template="plotly_white", opacity=0.6,
            )
            return (
                fig,
                f"Scatter: {x_col} vs {y_col}",
                f"Scatter of {x_col} vs {y_col} in {dataset_name}",
            )

        if chart_type == "correlation_heatmap":
            num = df.select_dtypes(include="number")
            if num.shape[1] < 2:
                return None, "", "Need ≥2 numeric columns."
            corr = num.corr()
            fig = px.imshow(
                corr, title=f"Correlation — {dataset_name}",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                text_auto=".2f", template="plotly_white",
            )
            return (
                fig,
                "Correlation Heatmap",
                f"Correlation of {num.shape[1]} numeric cols in {dataset_name}",
            )

        if chart_type == "box":
            col = params.get("column")
            if not col or col not in df.columns:
                log.warning(
                    "_build_figure | box: column=%r not in dataset=%s", col, dataset_name
                )
                return None, "", f"Column '{col}' not found."
            gb = params.get("group_by")
            fig = px.box(
                df, y=col, x=gb if gb and gb in df.columns else None,
                title=f"Box Plot: {col}", template="plotly_white",
            )
            label = f"Box: {col}" + (f" by {gb}" if gb and gb in df.columns else "")
            return fig, label, f"Box plot of {col} in {dataset_name}"

        if chart_type == "violin":
            col = params.get("column")
            if not col or col not in df.columns:
                log.warning(
                    "_build_figure | violin: column=%r not in dataset=%s", col, dataset_name
                )
                return None, "", f"Column '{col}' not found."
            if not pd.api.types.is_numeric_dtype(df[col]):
                log.warning(
                    "_build_figure | violin: column=%r is not numeric in dataset=%s", col, dataset_name
                )
                return None, "", f"Column '{col}' must be numeric for a violin plot."
            gb = params.get("group_by")
            fig = px.violin(
                df, y=col,
                x=gb if gb and gb in df.columns else None,
                box=True, points="outliers",
                title=f"Violin: {col}", template="plotly_white",
            )
            label = f"Violin: {col}" + (f" by {gb}" if gb and gb in df.columns else "")
            return fig, label, f"Violin of {col} in {dataset_name}"

        if chart_type == "pairplot":
            requested = params.get("columns") or []
            # Keep only existing, numeric columns (scatter matrix requires numeric).
            cols = [
                c for c in requested
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ][:4]
            if len(cols) < 2:
                # Fall back: auto-pick the first 4 numeric columns.
                cols = df.select_dtypes(include="number").columns.tolist()[:4]
            if len(cols) < 2:
                return None, "", "pairplot needs ≥2 numeric columns."

            color_col = params.get("color_column")
            valid_color = color_col if color_col and color_col in df.columns else None
            # Build data slice without duplicating the color column.
            plot_cols = list(dict.fromkeys(cols + ([valid_color] if valid_color and valid_color not in cols else [])))
            plot_df = df.head(2000)[plot_cols].copy()

            fig = px.scatter_matrix(
                plot_df,
                dimensions=cols,
                color=valid_color,
                title=f"Pairplot: {', '.join(cols)}",
                template="plotly_white",
            )
            # Hide upper triangle and diagonal to reduce visual clutter.
            fig.update_traces(
                diagonal=dict(visible=False),
                showupperhalf=False,
            )
            return (
                fig,
                f"Pairplot ({len(cols)} cols)",
                f"Pairplot of {cols} in {dataset_name}",
            )

        if chart_type == "missing_heatmap":
            mask = df.isnull()
            missing_cols = mask.columns[mask.any()].tolist()
            if not missing_cols:
                return None, "", "No missing values — nothing to chart."
            sample = mask[missing_cols].head(100).astype(int)
            fig = px.imshow(
                sample.T,
                title=f"Missing Values — {dataset_name} (first 100 rows)",
                color_continuous_scale=[[0, "#f0f0f0"], [1, "#e53e3e"]],
                labels={"x": "Row", "y": "Column", "color": "Missing"},
                template="plotly_white",
            )
            return (
                fig,
                "Missing Heatmap",
                f"Missing pattern in {len(missing_cols)} cols of {dataset_name}",
            )

        log.warning("_build_figure | unknown chart_type=%r for dataset=%s", chart_type, dataset_name)
        return None, "", f"Unknown chart_type: {chart_type}"
    except Exception as exc:
        log.exception("_build_figure | exception chart_type=%r dataset=%s", chart_type, dataset_name)
        return None, "", f"Chart error: {exc}"
