"""Phase B: ledger gate, exit invariant, wait_for_training, notes, resilience, budget counting."""

import json

import pytest
from fakes import FakeBackboard, call, text
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from test_approval import settings as approval_settings
from test_domain import GOOD

from posttrain_agent.agent import ORCHESTRATOR_PROMPT, build_agent
from posttrain_agent.config import Settings
from posttrain_agent.experiments import ledger as ledger_mod
from posttrain_agent.experiments import runstate as rs
from posttrain_agent.experiments.domain import LocalStubBackend, make_tools
from posttrain_agent.llm.resilience import SafeToolRetryMiddleware, build_model
from posttrain_agent.observability.events import CURRENT_RUN, Journal
from posttrain_agent.runtime.budget_guard import BudgetGuard, RunTokenLedger
from posttrain_agent.runtime.exit_gate import ExitGate, NotesInjector, exit_problem
from posttrain_agent.runtime.persistence import open_runtime
from posttrain_agent.runtime.replay import run_with_recovery
from posttrain_agent.subagents import ROLES, load_prompt


def R(score, split="heldout", **kw):
    return {"score": score, "evidence": "LIVE", "suite": "s", "split": split, "seed": 7, "n": 400,
            "parse_rate": 1.0, "judged_by": "j", "served_by": "v", "tasks_sha256": f"t-{split}", **kw}


class Jobs(LocalStubBackend):
    """Scripted provider: job_status walks through ``statuses``; evaluate is never needed."""

    def __init__(self, statuses, checkpoint="ckpt-1", tokens=1000):
        self.statuses, self.checkpoint, self.tokens, self.trains = list(statuses), checkpoint, tokens, 0

    def train(self, config, dataset_path, base_model):
        self.trains += 1
        return {"evidence": "LIVE", "status": "queued", "provider_job_id": f"ftjob-{self.trains}",
                "checkpoint": None}

    def job_status(self, job_id):
        st = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return {"evidence": "LIVE", "provider_job_id": job_id, "status": st,
                "trained_tokens": self.tokens,
                "checkpoint": self.checkpoint if st == "succeeded" else None}


def mk(tmp_path, backend, **kw):
    s = approval_settings(tmp_path, **kw)
    sleeps: list[float] = []
    return make_tools(backend, s, sleep=sleeps.append), s, sleeps


def put(s, ckpt, split, result, status="ok", job=None):
    ledger_mod.append(s.state_dir, run="r", checkpoint=ckpt, split=split, result=result, job=job)


# ---- (1) run_gate -------------------------------------------------------------------------

def test_run_gate_reads_ledger_and_requires_regression(tmp_path):
    t, s, _ = mk(tmp_path, Jobs(["succeeded"]))
    for ck, sc, rg in (("base", 0.40, 0.90), ("ckA", 0.60, 0.90)):
        put(s, ck, "heldout", R(sc))
        put(s, ck, "regression", R(rg, "regression"))
    v = t["run_gate"]("base", "ckA")
    assert v["decision"] == "PROMOTE" and v["regression_checked"] is True
    # a regression-split drop blocks promotion even though held-out improved
    put(s, "ckB", "heldout", R(0.60))
    put(s, "ckB", "regression", R(0.50, "regression"))
    assert t["run_gate"]("base", "ckB")["decision"] == "REJECT"


def test_run_gate_missing_split_is_an_error_not_a_verdict(tmp_path):
    t, s, _ = mk(tmp_path, Jobs(["succeeded"]))
    put(s, "base", "heldout", R(0.4))
    put(s, "ckA", "heldout", R(0.6))
    out = t["run_gate"]("base", "ckA")
    assert out["stage"] == "ledger" and "candidate regression" in out["missing"]
    assert "decision" not in out


# ---- (3) wait_for_training, run state, duplicate launch ----------------------------------

def launch(t, hp=GOOD):
    return t["launch_training"](hp, "d.jsonl")


def test_wait_for_training_blocks_records_state_and_reconciles_tokens(tmp_path):
    t, s, sleeps = mk(tmp_path, Jobs(["running", "running", "succeeded"], tokens=4321))
    job = launch(t)["provider_job_id"]
    out = t["wait_for_training"](job, timeout_s=300, poll_s=5)
    assert out["status"] == "succeeded" and out["terminal"] and out["checkpoint"] == "ckpt-1"
    assert sleeps == [5, 5]
    rec = rs.find_job(s.state_dir, job)
    assert rec["status"] == "succeeded" and rec["checkpoint"] == "ckpt-1"
    assert rec["trained_tokens"] == 4321 and rec["dataset_sha256"] and rec["hparams"] == GOOD
    spend = [json.loads(x) for x in (s.state_dir / "spend.jsonl").read_text().splitlines()]
    assert any(x.get("trained_tokens") == 4321 for x in spend)


def test_wait_times_out_and_succeeded_without_checkpoint_is_not_success(tmp_path):
    t, s, _ = mk(tmp_path, Jobs(["running"]))
    out = t["wait_for_training"](launch(t)["provider_job_id"], timeout_s=10, poll_s=5)
    assert out["timed_out"] and not out["terminal"]
    t2, _, _ = mk(tmp_path / "x", Jobs(["succeeded"], checkpoint=None))
    out = t2["wait_for_training"](launch(t2)["provider_job_id"])
    assert out["status"] == "FAILED_NO_CHECKPOINT" and "no checkpoint" in out["error"]


def test_duplicate_launch_refused_unless_earlier_job_terminal_failed(tmp_path):
    be = Jobs(["running", "failed"])
    t, s, _ = mk(tmp_path, be)
    first = launch(t)
    again = launch(t)
    assert again["status"] == "BLOCKED" and again["stage"] == "duplicate"
    assert again["provider_job_id"] == first["provider_job_id"] and be.trains == 1
    t["wait_for_training"](first["provider_job_id"], timeout_s=60, poll_s=1)  # -> failed
    assert launch(t)["provider_job_id"] == "ftjob-2"  # terminal-failed: relaunch allowed


def test_failed_launch_call_does_not_block_a_retry(tmp_path):
    class Boom(Jobs):
        def train(self, *a):
            raise RuntimeError("net")
    t, s, _ = mk(tmp_path, Boom(["running"]))
    with pytest.raises(RuntimeError):
        launch(t)
    assert rs.run_states(s.state_dir)[-1]["status"] == "LAUNCH_FAILED"


def test_evaluate_candidate_resolves_from_run_state(tmp_path):
    seen = []

    class B(Jobs):
        def evaluate(self, checkpoint, suite, split, seed):
            seen.append(checkpoint)
            return {"evidence": "LIVE", "score": 0.5, "checkpoint": checkpoint}

    t, s, _ = mk(tmp_path, B(["succeeded"]))
    assert t["evaluate_checkpoint"]("candidate")["stage"] == "candidate"
    job = launch(t)["provider_job_id"]
    t["wait_for_training"](job)
    assert t["evaluate_checkpoint"]("candidate", "heldout")["score"] == 0.5
    assert seen == ["ckpt-1"]
    assert ledger_mod.latest_ok(s.state_dir, job=job, split="heldout")


# ---- (2) exit gate --------------------------------------------------------------------------

def seed_candidate(s, status="succeeded", ckpt="ckpt-1", run="r"):
    rs.write_run_state(s.state_dir, run=run, key="k", job_id="ftjob-1", status=status,
                       checkpoint=ckpt if status == "succeeded" else None)


def test_exit_problem_walks_the_missing_steps(tmp_path):
    s = Settings(state_dir=tmp_path)
    assert exit_problem(s.state_dir, "r") is None  # nothing launched: no obligation
    seed_candidate(s, "running")
    assert "wait_for_training" in exit_problem(s.state_dir, "r") and "ftjob-1" in exit_problem(s.state_dir, "r")
    seed_candidate(s)
    assert "no valid evaluation" in exit_problem(s.state_dir, "r")
    ledger_mod.append(s.state_dir, run="r", checkpoint="ckpt-1", job="ftjob-1", split="heldout",
                      result={"error": "boom", "stage": "judge"})
    msg = exit_problem(s.state_dir, "r")
    assert "boom" in msg and "re-dispatch the evaluator with the job id ftjob-1" in msg
    for sp in ("heldout", "regression"):
        put(s, "ckpt-1", sp, R(0.5, sp))
    assert "no gate verdict" in exit_problem(s.state_dir, "r")
    rs.record_gate(s.state_dir, run="r", champion="base", candidate="ckpt-1", candidate_job="ftjob-1",
                   decision="REJECT")
    assert exit_problem(s.state_dir, "r") is None


def test_abort_experiment_satisfies_the_invariant(tmp_path):
    t, s, _ = mk(tmp_path, Jobs(["failed"]))
    token = CURRENT_RUN.set("r")
    try:
        seed_candidate(s, "failed")
        assert exit_problem(s.state_dir, "r")
        assert "error" in t["abort_experiment"]("  ")
        assert t["abort_experiment"]("training failed")["aborted"]
        assert exit_problem(s.state_dir, "r") is None
    finally:
        CURRENT_RUN.reset(token)


@tool
def ping() -> str:
    """pong"""
    return "pong"


def _graph(script, log, state_dir, **gate):
    return create_agent(build_model(Settings(), client_factory=lambda: FakeBackboard(script, log)),
                        tools=[ping], middleware=[ExitGate(state_dir, **gate)])


async def test_exit_gate_reprompts_max_three_times_then_lets_the_run_end(tmp_path):
    seed_candidate(Settings(state_dir=tmp_path), "running")
    log: list = []
    token = CURRENT_RUN.set("r")
    try:
        out = await _graph(lambda kw: text("done"), log, tmp_path).ainvoke(
            {"messages": [("user", "go")]}, {"recursion_limit": 50})
    finally:
        CURRENT_RUN.reset(token)
    assert len(log) == 4  # 1 + 3 re-prompts, then it may end
    prompts = [m.content for m in out["messages"] if "[exit gate" in str(m.content)]
    assert len(prompts) == 3 and "ftjob-1" in prompts[0] and "3/3" in prompts[2]


async def test_exit_gate_lets_a_run_finish_once_abort_is_recorded(tmp_path):
    s = Settings(state_dir=tmp_path)
    seed_candidate(s, "running")
    seq = [text("done"), text("done again")]
    log: list = []

    def script(kw):
        if len(log) == 2:  # operator-side effect between turns: the abort lands
            rs.record_abort(tmp_path, "r", "give up")
        return seq.pop(0)

    token = CURRENT_RUN.set("r")
    try:
        out = await _graph(script, log, tmp_path).ainvoke({"messages": [("user", "go")]})
    finally:
        CURRENT_RUN.reset(token)
    assert len(log) == 2 and out["messages"][-1].content == "done again"


async def test_replay_marks_incomplete_when_the_invariant_does_not_hold(tmp_path):
    s = approval_settings(tmp_path)
    seed_candidate(s, "running", run="inc")
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=lambda: FakeBackboard(
            lambda kw: text("final report"), []))
        rep = await run_with_recovery(
            graph, "go", "inc", max_attempts=1,
            exit_check=lambda th: exit_problem(s.state_dir, th))
    assert rep.status == "INCOMPLETE" and "ended without a gate verdict" in rep.errors[-1]


async def test_general_purpose_subagent_is_disabled(tmp_path):
    s = approval_settings(tmp_path)
    seq = [call("task", {"description": "x", "subagent_type": "general-purpose"}), text("ok")]
    log: list = []
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=lambda: FakeBackboard(
            lambda kw: seq.pop(0), log))
        rep = await run_with_recovery(graph, "go", "gp", max_attempts=1)
    assert rep.status == "COMPLETED"
    msgs = rep.result["messages"]
    assert any("general-purpose is disabled" in str(m.content) for m in msgs)
    assert len(log) == 2  # the disabled subagent never ran a model call


# ---- (4) prompts ----------------------------------------------------------------------------

def test_coordinator_prompt_rules():
    p = ORCHESTRATOR_PROMPT
    for needle in ("Termination rule", "abort_experiment", "BLOCKED", "FAILED", "job id",
                   "[operator note]", "input hashes", "at most 2 subagents", "never more than 2",
                   "general-purpose"):
        assert needle in p, needle


def test_role_prompts_match_the_new_tools():
    ex, ev, cm = (load_prompt(ROLES[r][0]) for r in ("training_executor", "evaluator",
                                                      "champion_manager"))
    assert "wait_for_training" in ex and "15 seconds" not in ex and "get_training_status" not in ex
    assert "requires a non-null `checkpoint`" in ex
    assert "VERBATIM" in ev and "by_game" in ev and "`candidate`" in ev
    assert "run_gate" in cm and "decide_promotion" not in cm and "regression" in cm
    assert "get_training_status" not in ROLES["training_executor"][2]
    assert ROLES["champion_manager"][2] == ("run_gate",)
    assert max(len(line) for line in ev.splitlines()) < 140


# ---- (5) resilience -------------------------------------------------------------------------

def test_task_and_launch_are_never_tool_retried():
    mw = SafeToolRetryMiddleware(max_retries=1)
    assert not mw._should_retry_tool("task") and not mw._should_retry_tool("launch_training")
    assert mw._should_retry_tool("web_search")


class FlakyGraph:
    """Minimal graph: raises ``errors`` in order, then returns."""

    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    async def aget_state(self, cfg):
        class S:
            values, next, tasks, config, metadata = {"m": 1}, (), (), {}, {}
        return S()

    async def ainvoke(self, payload, cfg):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return {"messages": [AIMessage(content="ok")]}


async def test_outage_is_waited_out_with_jittered_backoff_and_retrying_events(tmp_path):
    from backboard import BackboardServerError
    sleeps: list[float] = []

    async def fake_sleep(d):
        sleeps.append(d)

    j = Journal(tmp_path / "e.sqlite")
    g = FlakyGraph([BackboardServerError("down", 503)] * 6)
    rep = await run_with_recovery(g, "go", "o", max_attempts=2, base_delay=2, journal=j,
                                  sleep=fake_sleep, jitter=lambda: 0.5)
    assert rep.status == "COMPLETED" and g.calls == 7 and len(sleeps) == 6  # > max_attempts
    assert sleeps[:3] == [2.0, 4.0, 8.0]
    statuses = [e.data["status"] for e in j.read("o") if e.type == "run.status"]
    assert statuses.count("RETRYING") == 6


async def test_outage_past_the_window_is_failed_transient_and_resumable(tmp_path):
    from backboard import BackboardServerError

    async def fake_sleep(d):
        pass

    g = FlakyGraph([BackboardServerError("down", 503)] * 50)
    rep = await run_with_recovery(g, "go", "o", base_delay=2, sleep=fake_sleep, jitter=lambda: 1,
                                  outage_budget_s=100)
    assert rep.status == "FAILED_TRANSIENT"
    # a non-transient error keeps the small budget
    g2 = FlakyGraph([ValueError("bug")] * 10)
    rep2 = await run_with_recovery(g2, "go", "p", max_attempts=2, sleep=fake_sleep)
    assert rep2.status == "FAILED" and g2.calls == 2


async def test_cli_note_stores_and_run_parser_has_auto_resume(monkeypatch, tmp_path, capsys):
    import argparse

    from posttrain_agent import cli
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda c: Settings(state_dir=tmp_path)))
    ns = argparse.Namespace(cmd="note", thread="t1", text="use lr 1e-4")
    assert await cli._main(ns) == 0
    assert json.loads(capsys.readouterr().out)["stored"] is True
    assert [n["text"] for n in rs.unread_notes(tmp_path, "t1")] == ["use lr 1e-4"]
    assert await cli._main(argparse.Namespace(cmd="note", thread="t1", text=" ")) == 2
    monkeypatch.setattr("sys.argv", ["posttrain", "run", "--auto-resume", "--thread", "z", "goal"])
    captured = {}
    monkeypatch.setattr(cli.asyncio, "run", lambda coro: (coro.close(), captured.setdefault("ok", 1))[1])
    monkeypatch.setattr(cli, "load_dotenv", lambda: 0)
    with pytest.raises(SystemExit):
        cli.main()
    assert captured["ok"] == 1


# ---- (6) notes ------------------------------------------------------------------------------

def test_notes_are_delivered_once_per_thread(tmp_path):
    rs.add_note(tmp_path, "t1", "prefer batch 4")
    rs.add_note(tmp_path, "t2", "other thread")
    assert [n["text"] for n in rs.unread_notes(tmp_path, "t1")] == ["prefer batch 4"]
    assert rs.unread_notes(tmp_path, "t1") == []
    rs.add_note(tmp_path, "t1", "second")
    assert [n["text"] for n in rs.unread_notes(tmp_path, "t1")] == ["second"]
    assert len(rs.unread_notes(tmp_path, "t2")) == 1


async def test_notes_injected_once_and_journaled_without_the_text(tmp_path):
    j = Journal(tmp_path / "e.sqlite")
    rs.add_note(tmp_path, "nt", "SECRET-LOOKING guidance text")
    seq = [call("ping", {}), text("done")]
    log: list = []
    agent = create_agent(
        build_model(Settings(), client_factory=lambda: FakeBackboard(lambda kw: seq.pop(0), log)),
        tools=[ping], middleware=[NotesInjector(tmp_path, j)])
    token = CURRENT_RUN.set("nt")
    try:
        out = await agent.ainvoke({"messages": [("user", "go")]})
    finally:
        CURRENT_RUN.reset(token)
    notes = [m for m in out["messages"] if "[operator note]" in str(m.content)]
    assert len(notes) == 1 and "guidance text" in notes[0].content
    ev = [e for e in j.read("nt") if e.type == "operator.note"]
    assert len(ev) == 1 and "text" not in ev[0].data and ev[0].data["text_len"] > 0


# ---- (7) budget guard is linear ---------------------------------------------------------------

async def test_budget_guard_counts_only_new_messages(tmp_path):
    ledger = RunTokenLedger(tmp_path / "t.json", 10**9)
    n = {"i": 0}

    def script(kw):
        n["i"] += 1
        return call("ping", {}, id=f"c{n['i']}") if n["i"] < 8 else text("done")

    agent = create_agent(
        build_model(Settings(), client_factory=lambda: FakeBackboard(script, [])), tools=[ping],
        middleware=[BudgetGuard("x", 50, 10**9, ledger)])
    token = CURRENT_RUN.set("lin")
    try:
        out = await agent.ainvoke({"messages": [("user", "go " * 400)]}, {"recursion_limit": 100})
    finally:
        CURRENT_RUN.reset(token)
    from posttrain_agent.runtime.budget_guard import estimate_tokens
    once = estimate_tokens(out["messages"])
    # every message counted about once (the quadratic version would be several times larger)
    assert once * 0.8 <= ledger.total("lin") <= once * 1.2


def test_fold_logs_operator_note_without_changing_state():
    from posttrain_agent.observability.fold import empty_state, fold
    st = fold(empty_state(), {"seq": 1, "ts": 1.0, "type": "operator.note",
                              "data": {"note_id": 3, "text_len": 9}})
    assert st["log"][-1]["summary"] == "note_id=3" and st["last_seq"] == 1
    assert {**st, "log": [], "last_seq": 0} == empty_state()
