from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping


def build_markdown_report(
    project_name: str,
    dataset_info: Mapping[str, Any],
    training_metadata: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> str:
    leaderboard = training_metadata.get("leaderboard", [])
    metrics = training_metadata.get("best_metrics", {})
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# {project_name} Model Report",
        "",
        f"Generated: {generated_at}",
        "",
        "## Dataset",
        "",
        f"- Source: {dataset_info.get('name', 'Unknown')}",
        f"- Rows: {profile.get('row_count', 'Unknown')}",
        f"- Columns: {profile.get('column_count', 'Unknown')}",
        f"- Missing cells: {profile.get('missing_cells', 'Unknown')}",
        f"- Duplicate rows: {profile.get('duplicate_rows', 'Unknown')}",
        "",
        "## Training Run",
        "",
        f"- Run ID: {training_metadata.get('run_id')}",
        f"- Task type: {training_metadata.get('task_type')}",
        f"- Target: {training_metadata.get('target_column')}",
        f"- Best model: {training_metadata.get('best_model_name')}",
        f"- Training rows: {training_metadata.get('row_count')}",
        "",
        "## Best Metrics",
        "",
    ]

    for metric, value in metrics.items():
        lines.append(f"- {metric}: {_format_metric(value)}")

    lines.extend(["", "## Leaderboard", ""])
    if leaderboard:
        lines.append("| Rank | Model | Status | Metrics |")
        lines.append("| ---: | --- | --- | --- |")
        for entry in leaderboard:
            metric_text = ", ".join(
                f"{key}: {_format_metric(value)}" for key, value in entry.get("metrics", {}).items()
            )
            rank = entry.get("rank") or ""
            lines.append(
                f"| {rank} | {entry.get('model', '')} | {entry.get('status', '')} | {metric_text} |"
            )
    else:
        lines.append("No leaderboard entries were recorded.")

    lines.extend(["", "## Features", ""])
    for column in training_metadata.get("feature_columns", []):
        lines.append(f"- {column}")

    return "\n".join(lines) + "\n"


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

