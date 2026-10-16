import json

import pytest
from fakes import FakeBackboard, call, text
from langchain.agents import create_agent
from langchain_core.tools import tool

from posttrain_agent.budget_guard import BudgetGuard
from posttrain_agent.config import Settings
from posttrain_agent.resilience import build_model


@tool
def ping() -> str:
    """Return pong."""
    return "pong"


def _agent(script, log, guard):
    factory = lambda: FakeBackboard(script, log)
    return create_agent(build_model(Settings(), client_factory=factory), tools=[ping],
                        middleware=[guard])


async def test_endless_tool_loop_is_stopped_at_call_cap():
    log: list = []
    agent = _agent(lambda kw: call("ping", {}), log, BudgetGuard("evaluator", 4, 10**9))
    out = await agent.ainvoke({"messages": [("user", "go")]}, {"recursion_limit": 100})
    final = json.loads(out["messages"][-1].content)
    assert final["status"] == "BUDGET_EXHAUSTED" and "call cap" in final["limit"]
    assert len(log) == 4  # exactly the cap, never a fifth model call


async def test_token_cap_stops_before_call_cap():
    log: list = []
    agent = _agent(lambda kw: call("ping", {}), log, BudgetGuard("evaluator", 50, 1))
    out = await agent.ainvoke({"messages": [("user", "go " * 200)]}, {"recursion_limit": 100})
    assert "token cap" in json.loads(out["messages"][-1].content)["limit"]
    assert len(log) == 1


async def test_model_is_told_to_wrap_up_near_the_cap_and_can_answer():
    seen: list[str] = []

    def script(kw):
        seen.append(json.dumps(kw["system"] if "system" in kw else kw, default=str))
        return text('{"status":"DONE"}') if len(seen) >= 5 else call("ping", {})

    log: list = []
    agent = _agent(script, log, BudgetGuard("evaluator", 5, 10**9))
    out = await agent.ainvoke({"messages": [("user", "go")]}, {"recursion_limit": 100})
    assert out["messages"][-1].content == '{"status":"DONE"}'
    assert "BUDGET NEARLY EXHAUSTED" in seen[4]  # 5th call = 4 of 5 used, past 75%
    assert "BUDGET NEARLY EXHAUSTED" not in "".join(seen[:3])


async def test_run_wide_ledger_caps_all_agents_and_survives_restart(tmp_path):
    from posttrain_agent.budget_guard import RunTokenLedger
    path = tmp_path / "tokens.json"
    log: list = []
    ledger = RunTokenLedger(path, max_tokens=1)
    guard = BudgetGuard("evaluator", 50, 10**9, ledger=ledger)
    out = await _agent(lambda kw: call("ping", {}), log, guard).ainvoke(
        {"messages": [("user", "go " * 100)]}, {"recursion_limit": 100})
    assert "run token cap" in json.loads(out["messages"][-1].content)["limit"]
    assert len(log) == 1
    # a new process reads the persisted total, so a resumed run cannot spend past the cap
    assert RunTokenLedger(path, max_tokens=1).total("") > 0


def test_rejects_non_positive_caps():
    with pytest.raises(ValueError):
        BudgetGuard("x", 0, 10)
