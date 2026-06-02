"""Multi-agent AutoML discovery system.

Architecture:
  AimlScientist (orchestrator) → delegates to specialist sub-agents:
    • EdaAgent
    • FeatureEngineeringAgent
    • ModelingAgent
    • ReviewAgent
    • FineTuningAgent
    • DriftAgent

All agents share a single AgentContext (notebook, datasets, runs) and yield
AutopilotStep objects up to the UI.
"""

from .base import (
    AgentContext,
    AutopilotStep,
    BaseAgent,
    build_azure_client,
    get_deployment,
)
from .drift_agent import DriftAgent
from .eda_agent import EdaAgent
from .feature_engineering_agent import FeatureEngineeringAgent
from .fine_tuning_agent import FineTuningAgent
from .modeling_agent import ModelingAgent
from .researcher_agent import ResearcherAgent
from .review_agent import ReviewAgent
from .scientist import AimlScientist

__all__ = [
    "AgentContext",
    "AimlScientist",
    "AutopilotStep",
    "BaseAgent",
    "DriftAgent",
    "EdaAgent",
    "FeatureEngineeringAgent",
    "FineTuningAgent",
    "ModelingAgent",
    "ResearcherAgent",
    "ReviewAgent",
    "build_azure_client",
    "get_deployment",
]
