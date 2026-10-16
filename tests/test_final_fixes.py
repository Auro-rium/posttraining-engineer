"""Final-review fixes: journal redaction, FAILED on fatal, CLI run binding, no post-launch re-read."""

from __future__ import annotations

import dataclasses
import sqlite3
from types import SimpleNamespace

import pytest

from posttrain_agent import cli
from posttrain_agent.experiments.domain import LocalStubBackend, make_tools
from posttrain_agent.llm.backboard_model import CassetteMiss
from posttrain_agent.observability.events import CURRENT_RUN, Journal
from posttrain_agent.runtime.replay import run_with_recovery
from posttrain_agent.tools import budget as budget_mod


class _Graph:
    def __init__(self, exc):
        self.exc = exc

    async def aget_state(self, cfg):
        return SimpleNamespace(values={}, config={}, metadata={})

    async def ainvoke(self, payload, cfg):
        raise self.exc


def _raw_rows(path):
    con = sqlite3.connect(path)
    try:
        return con.execute("select * from events").fetchall()
    finally:
        con.close()


async def test_exception_text_never_reaches_journal(tmp_path):
    j = Journal(tmp_path / "e.sqlite")
    rep = await run_with_recovery(_Graph(RuntimeError("CANARY-SECRET-123")), "g", "t",
                                  journal=j, max_attempts=2, base_delay=0)
    assert "CANARY-SECRET-123" in repr(rep.errors)  # full text stays in the report
    rows = _raw_rows(tmp_path / "e.sqlite")
    assert rows and "CANARY-SECRET-123" not in repr(rows)
    final = [e for e in j.read("t") if e.type == "run.status"][-1].data
    assert final["status"] == "FAILED" and final["errors"] == ["RuntimeError", "RuntimeError"]


async def test_fatal_marks_failed_in_journal_and_reraises(tmp_path):
    j = Journal(tmp_path / "e.sqlite")
    with pytest.raises(CassetteMiss):
        await run_with_recovery(_Graph(CassetteMiss("CANARY-SECRET-123")), "g", "t", journal=j)
    last = [e for e in j.read("t") if e.type == "run.status"][-1].data
    assert last == {"status": "FAILED", "attempt": 1, "errors": ["CassetteMiss"]}
    assert "CANARY-SECRET-123" not in repr(_raw_rows(tmp_path / "e.sqlite"))


def test_launch_does_not_reread_spend_after_paid_launch(tmp_path, monkeypatch, request):
    from test_approval import settings as approval_settings
    from test_domain import GOOD

    tok = CURRENT_RUN.set("r1")
    request.addfinalizer(lambda: CURRENT_RUN.reset(tok))

    j = Journal(tmp_path / "e.sqlite")
    s = approval_settings(tmp_path)
    tools = make_tools(LocalStubBackend(), s, journal=j)
    real = budget_mod.spent_usd
    calls = []
    monkeypatch.setattr(budget_mod, "spent_usd", lambda st: calls.append(1) or real(st))
    out = tools["launch_training"](GOOD, "d.jsonl")
    assert len(calls) == 1  # only the pre-launch estimate reads it
    est = out["estimate"]
    last = [e for e in j.read("r1") if e.type == "budget.update"][-1].data
    assert last["spent_usd"] == round(est["spent_usd"] + est["estimated_cost_usd"], 4)


async def test_cli_fork_binds_current_run(tmp_path, monkeypatch):
    from test_approval import settings as approval_settings

    s = dataclasses.replace(approval_settings(tmp_path), backboard_api_key="k")
    seen = {}

    async def fake_fork(graph, thread, ckpt, cbs=None):
        seen["run"] = CURRENT_RUN.get()
        return {"messages": [SimpleNamespace(content="ok")]}

    async def fake_build(*a, **k):
        return object()

    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda c: s))
    monkeypatch.setattr(cli, "build_agent", fake_build)
    monkeypatch.setattr(cli, "fork", fake_fork)
    before = CURRENT_RUN.get()
    args = SimpleNamespace(cmd="fork", thread="thr-9", checkpoint="c1")
    assert await cli._main(args) == 0
    assert seen["run"] == "thr-9" and CURRENT_RUN.get() == before
