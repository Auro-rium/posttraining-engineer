"""Focused enforcement tests for the live reasoning-agent contract."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from app.agents import (
    AGENT_KEYS,
    NEMOTRON_MODEL_ID,
    PROMPT_CONTRACT_VERSION,
    agent_prompt_metadata,
    get_prompt_contract,
    resolve_nemotron_model,
)
from app.agents.benchmark_agent import create_benchmark_agent
from app.agents.champion_manager_agent import create_champion_manager_agent
from app.agents.data_curator_agent import create_data_curator_agent
from app.agents.eval_agent import create_eval_agent
from app.agents.failure_analyst_agent import create_failure_analyst_agent
from app.agents.prompt_contract import PROMPT_LIBRARY_DIR
from app.agents.research_agent import create_research_agent
from app.agents.training_designer_agent import create_training_designer_agent
from app.agents.training_executor_agent import create_training_executor_agent
from app.runtime_config import RuntimeConfig

FACTORIES = (
    create_benchmark_agent,
    create_failure_analyst_agent,
    create_research_agent,
    create_data_curator_agent,
    create_training_designer_agent,
    create_training_executor_agent,
    create_eval_agent,
    create_champion_manager_agent,
)


def test_all_specialists_use_exact_nemotron_model() -> None:
    for factory in FACTORIES:
        specialist = factory()
        model = specialist.agent.model
        assert model.config["model_id"] == NEMOTRON_MODEL_ID
        assert specialist.prompt_metadata["model_id"] == NEMOTRON_MODEL_ID
        assert specialist.prompt_metadata["prompt_version"] == PROMPT_CONTRACT_VERSION


def test_model_override_fails_closed() -> None:
    with pytest.raises(ValueError, match="must use nvidia\.nemotron-super-3-120b"):
        create_research_agent(model="some.other.model")

    with pytest.raises(ValidationError, match="strands_model"):
        RuntimeConfig(_env_file=None, strands_model="some.other.model")


def test_prompt_contracts_are_explicit_hashed_and_safe() -> None:
    required_sections = (
        "MISSION",
        "INPUT CONTRACT",
        "OUTPUT CONTRACT",
        "EVIDENCE AND SAFETY RULES",
        "BOUNDED CREATIVITY",
        "sealed held-out",
        "status BLOCKED",
        "Never invent",
    )
    for agent_key in AGENT_KEYS:
        contract = get_prompt_contract(agent_key)
        prompt = contract.prompt
        assert all(section in prompt for section in required_sections)
        assert contract.prompt_sha256 == hashlib.sha256(prompt.encode()).hexdigest()
        assert agent_prompt_metadata(agent_key) == {
            "agent_key": agent_key,
            "model_id": NEMOTRON_MODEL_ID,
            "prompt_version": PROMPT_CONTRACT_VERSION,
            "prompt_sha256": contract.prompt_sha256,
            "prompt_file": contract.prompt_file,
        }


def test_contract_registry_has_exactly_eight_specialists() -> None:
    assert AGENT_KEYS == (
        "BenchmarkAgent",
        "FailureAnalystAgent",
        "ResearchAgent",
        "DataCuratorAgent",
        "TrainingDesignerAgent",
        "TrainingExecutorAgent",
        "EvalAgent",
        "ChampionManagerAgent",
    )


def test_prompt_library_has_one_reviewable_file_per_specialist() -> None:
    library_files = {
        path.name for path in PROMPT_LIBRARY_DIR.glob("*.md") if path.name != "README.md"
    }
    registered_files = {get_prompt_contract(agent_key).prompt_file for agent_key in AGENT_KEYS}
    assert library_files == registered_files
    assert len(library_files) == 8
    for agent_key in AGENT_KEYS:
        source = (PROMPT_LIBRARY_DIR / get_prompt_contract(agent_key).prompt_file).read_text()
        assert source.startswith(f"# {agent_key}\n")
        assert "held-out" in source.lower()


def test_model_resolver_accepts_only_pinned_id() -> None:
    assert resolve_nemotron_model() == NEMOTRON_MODEL_ID
    assert resolve_nemotron_model(NEMOTRON_MODEL_ID) == NEMOTRON_MODEL_ID
    with pytest.raises(ValueError):
        resolve_nemotron_model("nvidia.nemotron-super-3-120b:wrong-version")
