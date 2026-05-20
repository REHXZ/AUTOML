"""AI autopilot entry point — thin wrapper around the multi-agent system.

The heavy lifting now lives in `aiml_discovery.agents`:
  AimlScientist (orchestrator) → EdaAgent, FeatureEngineeringAgent,
  ModelingAgent, ReviewAgent, FineTuningAgent.

This module keeps the public `AiAutopilot` class so existing UI code in
`app.py` (and anything else) keeps working unchanged.
"""

from __future__ import annotations

from typing import Any, Generator

from .agents import (
    AgentContext,
    AimlScientist,
    AutopilotStep,
    build_azure_client,
    get_deployment,
)
from .storage import DatasetInfo, ProjectStore


class AiAutopilot:
    """Public façade that wires the AIML Scientist into the existing UI flow."""

    def __init__(
        self,
        api_key: str,
        project_id: str,
        store: ProjectStore,
        user_goal: str = "",
    ) -> None:
        self._client = build_azure_client(api_key)
        self._deployment = get_deployment()
        self._ctx = AgentContext(
            project_id=project_id,
            store=store,
            user_goal=user_goal,
        )
        self._scientist = AimlScientist(self._client, self._deployment, self._ctx)

    # ------------------------------------------------------------------
    # Generator surface used by app.py
    # ------------------------------------------------------------------

    def run(self) -> Generator[AutopilotStep, list[str] | None, None]:
        yield from self._scientist.run()

    # ------------------------------------------------------------------
    # Properties the UI reads after the run completes
    # ------------------------------------------------------------------

    @property
    def new_datasets(self) -> list[DatasetInfo]:
        return self._ctx.new_datasets

    @property
    def training_runs(self) -> list[dict[str, Any]]:
        return self._ctx.training_runs

    @property
    def strategy_summary(self) -> str:
        return self._scientist.strategy_summary

    @property
    def notebook(self) -> list[str]:
        return self._ctx.notebook


__all__ = ["AiAutopilot", "AutopilotStep"]
