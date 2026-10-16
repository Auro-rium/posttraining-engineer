import asyncio
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from posttrain_agent.observability.events import A2A_TEXT_MAX, Journal, sanitize
from posttrain_agent.runtime.concurrency import SubagentConcurrencyMiddleware


def _msg(text, **extra):
    return sanitize("a2a.message", {"from": "coordinator", "to": "evaluator", "kind": "brief",
                                    "text": text, **extra})


def test_text_is_stored_cut_and_its_true_length_kept():
    out = _msg("x" * 5000)
    assert len(out["text"]) == A2A_TEXT_MAX and out["text_len"] == 5000


def test_credentials_are_scrubbed(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "nb-realsecretvalue123")
    text = ("use nb-realsecretvalue123 and Authorization: Bearer abcdef123456 "
            "api_key=hunter2hunter2 also sk-abcdefghijklmnop1234 done")
    out = _msg(text)["text"]
    for leaked in ("nb-realsecretvalue123", "abcdef123456", "hunter2", "sk-abcdefghijklmnop1234"):
        assert leaked not in out
    assert out.count("[redacted]") >= 4 and out.endswith("done")


def test_unlisted_fields_and_non_string_text_are_dropped():
    assert "extra" not in _msg("hi", extra="no")
    assert "text" not in sanitize("a2a.message", {"from": "a", "to": "b", "text": {"x": 1}})


def test_brief_and_reply_are_journaled_for_task_calls(tmp_path):
    j = Journal(tmp_path / "e.sqlite")
    mw = SubagentConcurrencyMiddleware(limit=2, journal=j, run="r1")
    req = SimpleNamespace(tool_call={"name": "task", "args": {
        "subagent_type": "evaluator", "description": "Evaluate ft:abc on heldout"}})

    async def handler(r):
        return ToolMessage(content='{"status":"SUCCEEDED"}', tool_call_id="t1")

    asyncio.run(mw.awrap_tool_call(req, handler))
    msgs = [e.data for e in j.read("r1") if e.type == "a2a.message"]
    assert msgs[0] == {"from": "coordinator", "to": "evaluator", "kind": "brief",
                       "text": "Evaluate ft:abc on heldout", "text_len": 26}
    assert msgs[1]["from"] == "evaluator" and msgs[1]["kind"] == "result" and msgs[1]["ok"]
    assert msgs[1]["text"] == '{"status":"SUCCEEDED"}'
