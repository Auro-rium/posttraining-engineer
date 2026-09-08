"""Strands specialist agents for autonomous post-training."""

from .benchmark_agent import BenchmarkAgent, create_benchmark_agent
from .champion_manager_agent import ChampionManagerAgent, create_champion_manager_agent
from .data_curator_agent import DataCuratorAgent, create_data_curator_agent
from .eval_agent import EvalAgent, create_eval_agent
from .failure_analyst_agent import FailureAnalystAgent, create_failure_analyst_agent
from .prompt_contract import (
    AGENT_KEYS,
    NEMOTRON_MODEL_ID,
    PROMPT_CONTRACT_VERSION,
    AgentPromptContract,
    agent_prompt_metadata,
    get_prompt_contract,
    resolve_agent_model,
    resolve_nemotron_model,
)
from .research_agent import ResearchAgent, create_research_agent
from .training_designer_agent import TrainingDesignerAgent, create_training_designer_agent
from .training_executor_agent import TrainingExecutorAgent, create_training_executor_agent

__all__ = [
    "AGENT_KEYS",
    "NEMOTRON_MODEL_ID",
    "PROMPT_CONTRACT_VERSION",
    "AgentPromptContract",
    "BenchmarkAgent",
    "ChampionManagerAgent",
    "DataCuratorAgent",
    "EvalAgent",
    "FailureAnalystAgent",
    "ResearchAgent",
    "TrainingDesignerAgent",
    "TrainingExecutorAgent",
    "agent_prompt_metadata",
    "create_benchmark_agent",
    "create_champion_manager_agent",
    "create_data_curator_agent",
    "create_eval_agent",
    "create_failure_analyst_agent",
    "create_research_agent",
    "create_training_designer_agent",
    "create_training_executor_agent",
    "get_prompt_contract",
    "resolve_agent_model",
    "resolve_nemotron_model",
]
