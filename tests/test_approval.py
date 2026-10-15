"""Human-in-the-loop gate before training, through a real task -> subagent path."""

import json

from fakes import FakeBackboard, call, text
from test_domain import GOOD, ROW

from posttrain_agent import budget
from posttrain_agent.agent import build_agent
from posttrain_agent.config import Settings
from posttrain_agent.dataset import save_rows
from posttrain_agent.persistence import open_runtime
from posttrain_agent.replay import decide, ledger, pending_actions, run_with_recovery


def settings(tmp_path, **kw):
    s = Settings(state_dir=tmp_path / "state", data_dir=tmp_path / "state" / "data",
                 model_retries=0, price_per_mtok=1.0, **kw)
    save_rows(s.data_dir, "d.jsonl", [ROW])
    return s


def factory(seq, log):
    return lambda: FakeBackboard(lambda kw: seq.pop(0), log)


LAUNCH = [
    call("task", {"description": "launch the job", "subagent_type": "training_executor"}),
    call("launch_training", {"hyperparameters": GOOD, "dataset_ref": "d.jsonl"}, id="call_2"),
]


async def test_training_pauses_for_approval_then_runs_after_approve_in_new_process(tmp_path):
    s = settings(tmp_path)
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=factory(list(LAUNCH), []))
        rep = await run_with_recovery(graph, "go", "t", store=rt.store, max_attempts=1)
        assert rep.status == "AWAITING_APPROVAL"
        act = rep.pending[0]
        assert act["name"] == "launch_training" and "budget estimate" in act["description"]
        assert "estimated_cost_usd" in act["description"]
        assert budget.spent_usd(s) == 0  # nothing launched yet
        assert (await ledger(rt.store))[0]["status"] == "AWAITING_APPROVAL"

    # a different "process": new runtime, graph and client over the same SQLite files
    log: list = []
    seq = [text('{"status": "SUCCEEDED"}'), text("all done")]
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=factory(seq, log))
        assert len(await pending_actions(graph, "t")) == 1
        rep = await decide(graph, "t", True, store=rt.store, max_attempts=1)
        assert rep.status == "COMPLETED" and rep.result["messages"][-1].content == "all done"
        assert budget.spent_usd(s) > 0  # the job really launched, once
        sent = json.dumps(log)
        assert "stub-job-" in sent and "estimated_cost_usd" in sent


async def test_reject_blocks_launch_and_tells_the_agent(tmp_path):
    s = settings(tmp_path)
    log: list = []
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=factory(list(LAUNCH), []))
        await run_with_recovery(graph, "go", "t", store=rt.store, max_attempts=1)
    seq = [text('{"status": "REJECTED"}'), text("revised plan")]
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=factory(seq, log))
        rep = await decide(graph, "t", False, message="too expensive", store=rt.store, max_attempts=1)
        assert rep.status == "COMPLETED"
        assert budget.spent_usd(s) == 0
        assert "too expensive" in json.dumps(log)


async def test_decide_without_pending_action_is_an_error(tmp_path):
    import pytest

    s = settings(tmp_path)
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=factory([text("hi")], []))
        await run_with_recovery(graph, "go", "t", max_attempts=1)
        with pytest.raises(ValueError):
            await decide(graph, "t", True)


async def test_approval_can_be_disabled_for_unattended_runs(tmp_path):
    s = settings(tmp_path, require_approval=False)
    seq = [*LAUNCH, text('{"status":"SUCCEEDED"}'), text("done")]
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=factory(seq, []))
        rep = await run_with_recovery(graph, "go", "t", max_attempts=1)
        assert rep.status == "COMPLETED" and budget.spent_usd(s) > 0
