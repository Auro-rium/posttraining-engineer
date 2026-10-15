"""At most N subagents run at once; extra tasks queue instead of being dropped."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from fakes import FakeBackboard, text

from posttrain_agent.agent import build_agent
from posttrain_agent.concurrency import SubagentConcurrencyMiddleware
from posttrain_agent.config import Settings
from posttrain_agent.domain import LocalStubBackend
from posttrain_agent.persistence import open_runtime
from posttrain_agent.replay import run_with_recovery


def req(name):
    return SimpleNamespace(tool_call={"name": name})


async def test_limiter_caps_task_calls_but_not_other_tools():
    state = {"now": 0, "peak": 0}

    async def handler(_r):
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.02)
        state["now"] -= 1
        return "ok"

    mw = SubagentConcurrencyMiddleware(limit=2)
    results = await asyncio.gather(*(mw.awrap_tool_call(req("task"), handler) for _ in range(6)))
    assert results == ["ok"] * 6 and state["peak"] == 2 and mw.peak == 2  # queued, none dropped

    state["peak"] = 0
    await asyncio.gather(*(mw.awrap_tool_call(req("recall"), handler) for _ in range(5)))
    assert state["peak"] == 5  # only `task` is limited


def test_limit_must_be_positive():
    with pytest.raises(ValueError):
        SubagentConcurrencyMiddleware(0)


def test_sync_path_is_also_capped():
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def handler(_r):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.03)
        with lock:
            state["now"] -= 1
        return "ok"

    mw = SubagentConcurrencyMiddleware(limit=2)
    threads = [threading.Thread(target=mw.wrap_tool_call, args=(req("task"), handler)) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert state["peak"] == 2


class SlowBackend(LocalStubBackend):
    """Tracks real concurrency inside subagent tool execution."""

    def __init__(self):
        self.now, self.peak, self.lock = 0, 0, threading.Lock()

    def evaluate(self, checkpoint, suite, split, seed):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(0.15)
        with self.lock:
            self.now -= 1
        return super().evaluate(checkpoint, suite, split, seed)


def three_tasks():
    return {"content": "", "status": "REQUIRES_ACTION", "tool_calls": [
        {"id": f"t{i}", "type": "function", "function": {
            "name": "task",
            "arguments": '{"description": "evaluate base", "subagent_type": "evaluator"}'}}
        for i in range(3)]}


def evaluator_call():
    return {"content": "", "status": "REQUIRES_ACTION", "tool_calls": [
        {"id": "e1", "type": "function", "function": {
            "name": "evaluate_checkpoint", "arguments": '{"checkpoint": "base", "split": "heldout"}'}}]}


async def test_three_parallel_subagents_run_two_at_a_time(tmp_path):
    backend, limiter = SlowBackend(), SubagentConcurrencyMiddleware(2)
    served = {"subagent_calls": 0}

    def script(kw):
        coordinator = "You are the coordinator" in kw["system_prompt"]
        done = "tool_result" in kw["content"]
        if coordinator:
            return text("all evaluated") if done else three_tasks()
        served["subagent_calls"] += 1
        return text('{"status": "SUCCEEDED"}') if done else evaluator_call()

    s = Settings(state_dir=tmp_path / "s", data_dir=tmp_path / "s" / "d", model_retries=0)
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, experiments=backend, limiter=limiter,
                                  client_factory=lambda: FakeBackboard(script, []))
        rep = await run_with_recovery(graph, "go", "p", max_attempts=1)
    assert rep.status == "COMPLETED", rep.errors
    assert rep.result["messages"][-1].content == "all evaluated"
    assert served["subagent_calls"] == 6          # all three subagents ran to completion
    assert limiter.peak == 2                      # never more than two subagents at once
    assert backend.peak == 2                      # and two really did overlap (true parallelism)


async def test_two_paused_launches_are_both_answered_by_one_decision(tmp_path):
    import json

    from test_approval import settings as approval_settings
    from test_domain import GOOD

    from posttrain_agent import budget
    from posttrain_agent.replay import decide

    def two_executors():
        return {"content": "", "status": "REQUIRES_ACTION", "tool_calls": [
            {"id": f"x{i}", "type": "function", "function": {
                "name": "task", "arguments":
                '{"description": "launch", "subagent_type": "training_executor"}'}} for i in range(2)]}

    def launch():
        return {"content": "", "status": "REQUIRES_ACTION", "tool_calls": [
            {"id": "l1", "type": "function", "function": {
                "name": "launch_training", "arguments":
                json.dumps({"hyperparameters": GOOD, "dataset_ref": "d.jsonl"})}}]}

    def script(kw):
        coordinator = "You are the coordinator" in kw["system_prompt"]
        done = "tool_result" in kw["content"]
        if coordinator:
            return text("both launched") if done else two_executors()
        return text('{"status": "SUCCEEDED"}') if done else launch()

    s = approval_settings(tmp_path)
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=lambda: FakeBackboard(script, []))
        rep = await run_with_recovery(graph, "go", "q", max_attempts=1)
        assert rep.status == "AWAITING_APPROVAL" and len(rep.pending) == 2
        assert budget.spent_usd(s) == 0
        rep = await decide(graph, "q", True, max_attempts=1)
        assert rep.status == "COMPLETED", rep.errors
        assert rep.result["messages"][-1].content == "both launched"
        assert len(budget.spend_path(s).read_text().splitlines()) == 2  # both jobs launched
