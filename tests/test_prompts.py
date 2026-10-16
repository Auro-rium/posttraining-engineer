import json

import pytest

from posttrain_agent.agent import ORCHESTRATOR_PROMPT
from posttrain_agent.subagents import ROLES, load_prompt
from posttrain_agent.tools import dataset

SECTIONS = ("## MISSION", "## TOOLS YOU HAVE", "## PROCEDURE", "## HARD RULES", "## BUDGET",
            "## OUTPUT CONTRACT", "## BOUNDED CREATIVITY")


@pytest.mark.parametrize("role", ROLES)
def test_prompt_is_extensive_and_names_its_tools(role):
    file, _, tool_names, _ = ROLES[role]
    text = load_prompt(file)
    assert len(text.split()) >= 350
    for section in SECTIONS:
        assert section in text, f"{role} missing {section}"
    for tool in tool_names:
        assert f"`{tool}" in text, f"{role} prompt never mentions its tool {tool}"
    assert "FunctionGemma" not in text and "manifest_hash" not in text  # stale AWS-era terms


def test_coordinator_prompt_covers_budget_stops_and_resume():
    for needle in ("BUDGET_EXHAUSTED", "Resuming", "Final report", "held-out"):
        assert needle in ORCHESTRATOR_PROMPT


def test_every_role_and_the_coordinator_get_the_shared_project_definition():
    from posttrain_agent.subagents import build_subagents
    tools = {t: (lambda: None) for _, _, names, _ in ROLES.values() for t in names}
    for spec in build_subagents(tools):
        assert "PROJECT DEFINITION" in spec["system_prompt"], spec["name"]
        assert "supervised fine-tuning" in spec["system_prompt"]
    assert "not pretraining" in load_prompt("mission.md")


def test_researcher_playbook_uses_tavily_features_and_demands_sources():
    text = load_prompt("research.md")
    for needle in ("depth=\"advanced\"", "domains", "web_extract", "queries_run", "findings",
                   "source URL", "answer` text is not a source"):
        assert needle in text, needle


def test_game_tools_pass_the_dataset_verifier():
    row = {"messages": [
        {"role": "user", "content": "Nim: the pile has 6 stones."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "take_stones", "arguments": json.dumps({"count": 2})}}]}]}
    assert dataset.verify_rows([row])["ok"]
