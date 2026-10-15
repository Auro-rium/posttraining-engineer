"""End-to-end: real deepagents graph + AsyncSqlite persistence + fake Backboard."""

from backboard import BackboardValidationError
from fakes import FakeBackboard, call, text

from posttrain_agent.agent import build_agent
from posttrain_agent.config import Settings
from posttrain_agent.persistence import open_runtime
from posttrain_agent.replay import fork, history, ledger, run_with_recovery


def settings(tmp_path, **kw):
    return Settings(state_dir=tmp_path / "state", model_retries=0, **kw)


def factory(script, log):
    return lambda: FakeBackboard(script, log)


async def test_assistant_is_created_once_and_reused(tmp_path):
    log: list = []
    s = settings(tmp_path)
    async with open_runtime(s.state_dir) as rt:
        await build_agent(s, rt, client_factory=factory([], log))
        await build_agent(s, rt, client_factory=factory([], log))
    assert (s.state_dir / "backboard.json").exists()


async def test_full_turn_uses_memory_tool_and_pins_assistant(tmp_path):
    log: list = []
    seq = [call("recall", {"query": "lr"}), text("done")]
    s = settings(tmp_path)
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=factory(lambda kw: seq.pop(0), log))
        rep = await run_with_recovery(graph, "go", "t1", store=rt.store, max_attempts=1)
    assert rep.status == "COMPLETED"
    assert rep.result["messages"][-1].content == "done"
    llm_calls = [k for k in log if "content" in k]
    assert {k["assistant_id"] for k in llm_calls} == {"asst-123"}
    assert any(k.get("search_memories") == "lr" for k in log)
    # the file-memory tier is loaded into the system prompt
    assert "Standing knowledge" in llm_calls[0]["system_prompt"]


async def test_crash_then_resume_in_new_process_does_not_rerun_finished_tool(tmp_path):
    s = settings(tmp_path)
    log1: list = []
    seq1 = [call("recall", {"query": "lr"}), BackboardValidationError("boom", 400)]
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=factory(lambda kw: seq1.pop(0), log1))
        rep = await run_with_recovery(graph, "go", "t2", store=rt.store, max_attempts=1)
        assert rep.status == "FAILED" and "boom" in rep.errors[0]
        assert rep.last_checkpoint_id

    # "restart": brand-new runtime/graph/client over the same SQLite files
    log2: list = []
    seq2 = [text("recovered")]
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=factory(lambda kw: seq2.pop(0), log2))
        rep = await run_with_recovery(graph, "go", "t2", store=rt.store, max_attempts=2,
                                      base_delay=0)
        assert rep.status == "COMPLETED"
        assert rep.result["messages"][-1].content == "recovered"
        assert not any("search_memories" in k for k in log2)  # tool result came from checkpoint
        assert (await ledger(rt.store))[0]["status"] == "COMPLETED"


async def test_history_and_fork_replay_from_checkpoint(tmp_path):
    s = settings(tmp_path)
    log: list = []
    seq = [call("recall", {"query": "q"}), text("first"), text("replayed")]
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=factory(lambda kw: seq.pop(0), log))
        await run_with_recovery(graph, "go", "t3", max_attempts=1)
        rows = await history(graph, "t3")
        assert len(rows) >= 3
        # fork from the checkpoint right after the tool ran (next node = model)
        after_tool = next(r for r in rows if r["next"] == ["model"] and r["messages"] >= 3)
        out = await fork(graph, "t3", after_tool["checkpoint_id"])
        assert out["messages"][-1].content == "replayed"


async def test_filememory_persists_across_threads(tmp_path):
    s = settings(tmp_path)
    async with open_runtime(s.state_dir) as rt:
        await rt.backend.awrite("/memories/lessons.md", "lr 2e-4 regressed")
    async with open_runtime(s.state_dir) as rt:
        got = await rt.backend.aread("/memories/lessons.md")
        assert "regressed" in str(got)
