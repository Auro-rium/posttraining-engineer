from __future__ import annotations

import pytest

from app.adk_agents import ROLE_INSTRUCTIONS


@pytest.mark.parametrize("role", sorted(ROLE_INSTRUCTIONS))
def test_every_agent_prompt_has_complete_safety_contract(role: str) -> None:
    prompt = ROLE_INSTRUCTIONS[role]

    for clause in ("ROLE:", "GOAL:", "INPUTS:", "OUTPUT:", "PROCEDURE:"):
        assert clause in prompt
    assert "heldout" in prompt
    assert "secrets" in prompt
    assert "Fail closed" in prompt


@pytest.mark.parametrize(
    ("role", "required"),
    [
        ("benchmark_runner", "Never invent rewards"),
        ("failure_analyst", "reference only supplied trajectory IDs"),
        ("research_agent", "nonempty citations"),
        ("data_curator", "verifier alone admits SFTExample"),
        ("training_designer", "parameter allowlists"),
        ("training_executor", "Never ask Gemini to train"),
        ("evaluation_agent", "objective evaluator is authoritative"),
        ("champion_manager", "deterministic gate is authoritative"),
    ],
)
def test_each_agent_prompt_states_its_authoritative_boundary(role: str, required: str) -> None:
    assert required in ROLE_INSTRUCTIONS[role]
