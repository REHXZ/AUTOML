"""Multi-agent AutoML discovery system."""

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
from .hooks import Decision, Hook, HookContext, HookEvent, HookManager, HookOutcome
from .hook_policies import default_hook_manager
from .modeling_agent import ModelingAgent
from .researcher_agent import ResearcherAgent
from .review_agent import ReviewAgent
from .scientist import AimlScientist

__all__ = [
    # Core agent types
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
    # Hook lifecycle framework
    "Decision",
    "Hook",
    "HookContext",
    "HookEvent",
    "HookManager",
    "HookOutcome",
    "default_hook_manager",
]
