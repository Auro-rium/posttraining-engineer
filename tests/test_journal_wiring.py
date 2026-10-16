"""Model/tool callbacks and subagent start/stop are journaled, redacted, and throttled."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from posttrain_agent.observability.events import CURRENT_RUN, Journal
from posttrain_agent.observability.journal_handler import JournalHandler
from posttrain_agent.runtime.concurrency import SubagentConcurrencyMiddleware


@pytest.fixture(autouse=True)
def _run():
    tok = CURRENT_RUN.set("r1")
    yield
    CURRENT_RUN.reset(tok)


def _model_call(h, provider="bb", model="m", usage=None, error=None):
    rid = uuid4()
    h.on_chat_model_start({}, [[]], run_id=rid,
                          metadata={"ls_provider": provider, "ls_model_name": model})
    if error:
        h.on_llm_error(error, run_id=rid)
        return
    msg = SimpleNamespace(usage_metadata=usage or {"input_tokens": 3, "output_tokens": 4})
    resp = SimpleNamespace(generations=[[SimpleNamespace(message=msg)]])
    h.on_llm_end(resp, run_id=rid)


def test_model_and_tool_calls_are_journaled_without_prompt_text(tmp_path):
    j = Journal(tmp_path / "e.sqlite")
    h = JournalHandler(j, primary="bb:m")
    rid = uuid4()
    h.on_chat_model_start({}, [[SimpleNamespace(content="SECRET PROMPT")]], run_id=rid,
                          metadata={"ls_provider": "bb", "ls_model_name": "m"})
    resp = SimpleNamespace(generations=[[SimpleNamespace(
        text="SECRET COMPLETION",
        message=SimpleNamespace(usage_metadata={"input_tokens": 3, "output_tokens": 4}))]])
    h.on_llm_end(resp, run_id=rid)
    tid = uuid4()
    h.on_tool_start({"name": "recall"}, "SECRET ARGS", run_id=tid)
    h.on_tool_end("SECRET OUT", run_id=tid)
    tid = uuid4()
    h.on_tool_start({"name": "web_search"}, "x", run_id=tid)
    h.on_tool_error(ValueError("boom SECRET"), run_id=tid)
    ev = j.read("r1")
    assert [e.type for e in ev] == ["model.call", "tool.call", "tool.call"]
    m = ev[0].data
    assert m["name"] == "bb:m" and m["ok"] is True and m["fallback"] is False
    assert m["input_tokens"] == 3 and m["output_tokens"] == 4 and m["ms"] >= 0
    assert ev[1].data["name"] == "recall" and ev[1].data["ok"] is True
    assert ev[2].data["ok"] is False and ev[2].data["error"] == "ValueError"
    assert "SECRET" not in repr([e.data for e in ev])


def test_fallback_flag_set_when_model_differs_from_primary(tmp_path):
    j = Journal(tmp_path / "e.sqlite")
    h = JournalHandler(j, primary="bb:m")
    _model_call(h, model="m")
    _model_call(h, model="other")
    assert [e.data["fallback"] for e in j.read("r1")] == [False, True]


def test_no_primary_means_never_fallback(tmp_path):
    j = Journal(tmp_path / "e.sqlite")
    h = JournalHandler(j)
    _model_call(h, model="x")
    assert j.read("r1")[0].data["fallback"] is False


def test_snapshot_event_throttled_to_5s(tmp_path):
    j = Journal(tmp_path / "e.sqlite")
    now = [100.0]
    h = JournalHandler(j, primary="bb:m", clock=lambda: now[0])
    _model_call(h)
    now[0] = 102.0
    _model_call(h)
    assert [e.type for e in j.read("r1")].count("telemetry.snapshot") == 0
    now[0] = 105.5
    _model_call(h, error=RuntimeError("x"))
    snaps = [e for e in j.read("r1") if e.type == "telemetry.snapshot"]
    assert len(snaps) == 1
    d = snaps[0].data
    assert set(d) == {"p50_ms", "p95_ms", "error_rate", "input_tokens", "output_tokens"}
    assert d["input_tokens"] == 6 and d["output_tokens"] == 8
    assert d["error_rate"] == pytest.approx(1 / 3)
    now[0] = 107.0
    _model_call(h)
    assert len([e for e in j.read("r1") if e.type == "telemetry.snapshot"]) == 1
    now[0] = 111.0
    _model_call(h)
    assert len([e for e in j.read("r1") if e.type == "telemetry.snapshot"]) == 2


def test_run_id_resolved_at_emit_time(tmp_path):
    j = Journal(tmp_path / "e.sqlite")
    h = JournalHandler(j)
    CURRENT_RUN.set("late")
    _model_call(h)
    assert len(j.read("late")) == 1


def req(role):
    return SimpleNamespace(tool_call={"name": "task", "args": {
        "subagent_type": role, "description": "SECRET DESC"}})


async def test_two_parallel_subagents_get_slots_1_and_2_then_reuse(tmp_path):
    j = Journal(tmp_path / "e.sqlite")
    mw = SubagentConcurrencyMiddleware(limit=2, journal=j)
    gate = asyncio.Event()

    async def handler(r):
        if r.tool_call["args"]["subagent_type"] == "boom":
            raise RuntimeError("x")
        await gate.wait()
        return "ok"

    t1 = asyncio.create_task(mw.awrap_tool_call(req("evaluator"), handler))
    t2 = asyncio.create_task(mw.awrap_tool_call(req("trainer"), handler))
    await asyncio.sleep(0.05)
    gate.set()
    await asyncio.gather(t1, t2)
    await mw.awrap_tool_call(req("evaluator"), handler)
    with pytest.raises(RuntimeError):
        await mw.awrap_tool_call(req("boom"), handler)
    ev = [(e.type, e.data) for e in j.read("r1")]
    starts = [d for t, d in ev if t == "agent.start"]
    assert {(d["role"], d["slot"]) for d in starts[:2]} == {("evaluator", 1), ("trainer", 2)}
    assert starts[2] == {"role": "evaluator", "slot": 1}
    ends = [d for t, d in ev if t == "agent.end"]
    assert len(ends) == 4 and all(d["ok"] for d in ends[:3])
    assert ends[3] == {"role": "boom", "slot": 1, "ok": False}
    assert "SECRET" not in repr(ev)


def test_sync_path_journals_and_non_task_calls_do_not(tmp_path):
    j = Journal(tmp_path / "e.sqlite")
    mw = SubagentConcurrencyMiddleware(limit=2, journal=j)
    mw.wrap_tool_call(SimpleNamespace(tool_call={"name": "recall", "args": {}}), lambda r: 1)
    assert j.read("r1") == []
    mw.wrap_tool_call(req("evaluator"), lambda r: 1)
    assert [e.type for e in j.read("r1")] == ["agent.start", "agent.end"]


# --- Task 4: run-controller and domain-tool events ---------------------------------------

from fakes import text
from test_approval import LAUNCH, factory
from test_approval import settings as approval_settings

from posttrain_agent.agent import build_agent
from posttrain_agent.experiments.domain import LocalStubBackend, make_tools
from posttrain_agent.runtime.persistence import open_runtime
from posttrain_agent.runtime.replay import decide, parse_estimate_usd, run_with_recovery


def _is_subsequence(want, got):
    it = iter(got)
    return all(any(w == g for g in it) for w in want)


async def test_full_fake_run_emits_expected_event_sequence(tmp_path):
    s = approval_settings(tmp_path)
    j = Journal(tmp_path / "e.sqlite")
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=factory(list(LAUNCH), []), journal=j)
        rep = await run_with_recovery(graph, "go", "t", store=rt.store, max_attempts=1,
                                      journal=j, backend="stub", experiment_budget=3)
        assert rep.status == "AWAITING_APPROVAL"
    seq = [text('{"status": "SUCCEEDED"}'), text("all done")]
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=factory(seq, []), journal=j)
        rep = await decide(graph, "t", False, message="SECRET REASON", store=rt.store,
                           max_attempts=1, journal=j)
        assert rep.status == "COMPLETED"
    ev = j.read("t")
    marks = [("run.status", e.data.get("status")) if e.type == "run.status" else e.type
             for e in ev]
    assert _is_subsequence(
        ["run.started", ("run.status", "RUNNING"), "agent.start", "approval.pending",
         ("run.status", "AWAITING_APPROVAL"), "approval.resolved", ("run.status", "RUNNING"),
         ("run.status", "COMPLETED")], marks), marks
    assert "checkpoint" in marks
    started = next(e for e in ev if e.type == "run.started").data
    assert started == {"goal_len": 2, "experiment_budget": 3, "backend": "stub"}
    pend = next(e for e in ev if e.type == "approval.pending").data
    assert pend["action"] == "launch_training"
    assert pend["cli_hint"] == "posttrain approve --thread t"
    assert pend["dataset_ref"] == "d.jsonl" and isinstance(pend["estimate_usd"], float)
    res = next(e for e in ev if e.type == "approval.resolved").data
    assert res == {"decision": "reject", "message_len": len("SECRET REASON")}
    assert "SECRET REASON" not in repr([e.data for e in ev])


def test_estimate_parse_is_pinned():
    desc = ('Approve?\ndataset: d\nbudget estimate: {"estimated_cost_usd": 1.25, '
            '"within_budget": true}')
    assert parse_estimate_usd(desc) == 1.25
    assert parse_estimate_usd('budget estimate: {"estimated_cost_usd": null}') is None
    assert parse_estimate_usd('budget estimate: {"error": "cannot read dataset"}') is None
    assert parse_estimate_usd("budget estimate: not json") is None
    assert parse_estimate_usd("no estimate here") is None
    assert parse_estimate_usd(None) is None


def test_score_event_carries_evidence_label(tmp_path):
    j = Journal(tmp_path / "e.sqlite")
    s = approval_settings(tmp_path)
    tools = make_tools(LocalStubBackend(), s, journal=j)
    tools["run_benchmark"]("base")
    tools["evaluate_checkpoint"]("stub-ckpt-1", "heldout")
    sc = [e.data for e in j.read("r1") if e.type == "score"]
    assert [d["split"] for d in sc] == ["train", "heldout"]
    assert all(d["evidence"] == "EXPLANATION" and isinstance(d["value"], float) for d in sc)
    assert sc[1]["checkpoint"] == "stub-ckpt-1"


def test_domain_tools_emit_budget_job_gate_and_log_events(tmp_path):
    from test_domain import GOOD

    j = Journal(tmp_path / "e.sqlite")
    s = approval_settings(tmp_path)
    tools = make_tools(LocalStubBackend(), s, journal=j)
    tools["estimate_budget"]("d.jsonl", GOOD)
    out = tools["launch_training"](GOOD, "d.jsonl")
    tools["get_training_status"](out["provider_job_id"])
    tools["read_training_logs"](out["provider_job_id"])
    tools["decide_promotion"]({"score": 0.3}, {"score": 0.5})
    ev = j.read("r1")
    assert [e.type for e in ev] == ["budget.update", "budget.update", "job.update", "job.update",
                                    "job.log", "gate.verdict"]
    assert ev[0].data["price_known"] is True and ev[0].data["cap_usd"] == s.max_cost_usd
    assert ev[3].data["status"] == "succeeded"
    assert ev[4].data["message"] == "simulated job; no real logs exist"


async def test_journal_failure_does_not_fail_the_run(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    bad = Journal(blocker / "sub" / "e.sqlite")  # parent is a file: unwritable
    s = approval_settings(tmp_path, require_approval=False)
    seq = [*LAUNCH, text('{"status":"SUCCEEDED"}'), text("done")]
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=factory(seq, []), journal=bad)
        rep = await run_with_recovery(graph, "go", "t", max_attempts=1, journal=bad)
        assert rep.status == "COMPLETED"
    tools = make_tools(LocalStubBackend(), s, journal=bad)
    assert "score" in tools["run_benchmark"]("base")


def test_sample_journal_has_every_event_type_and_only_simulated_evidence(tmp_path):
    import json
    import sys
    from pathlib import Path

    from posttrain_agent.observability.events import EVENT_TYPES

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from make_sample_journal import make_sample

    out = tmp_path / "sample-run.jsonl"
    make_sample(out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {"seq", "ts", "run", "type", "data"} == set(rows[0])
    seen = {r["type"] for r in rows}
    assert not (EVENT_TYPES - {"job.log", "telemetry.snapshot"} - seen), EVENT_TYPES - seen
    assert seen <= EVENT_TYPES
    assert all(r["run"].startswith("sample") for r in rows)
    evid = [r["data"]["evidence"] for r in rows if "evidence" in r["data"]]
    assert evid and set(evid) == {"EXPLANATION"}
    assert {r["data"]["name"] for r in rows if r["type"] == "model.call"} == {
        "sample:simulated-model"}
    again = tmp_path / "again.jsonl"
    make_sample(again)
    assert again.read_text() == out.read_text()  # deterministic
