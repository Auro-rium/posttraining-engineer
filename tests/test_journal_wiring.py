"""Model/tool callbacks and subagent start/stop are journaled, redacted, and throttled."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from posttrain_agent.concurrency import SubagentConcurrencyMiddleware
from posttrain_agent.events import CURRENT_RUN, Journal
from posttrain_agent.journal_handler import JournalHandler


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
