import pytest
from backboard import BackboardServerError
from fakes import FakeBackboard, call, text
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from posttrain_agent.backboard_model import Cassette, CassetteMiss, ChatBackboard


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def model(script, log=None, **kw):
    return ChatBackboard(
        assistant_id="asst-1", client_factory=lambda: FakeBackboard(script, log), **kw
    )


async def test_text_reply_is_stateless_and_pins_assistant():
    log: list = []
    m = model([text("hi")], log)
    out = await m.ainvoke([SystemMessage("be brief"), HumanMessage("hello")])
    assert out.content == "hi"
    sent = log[0]
    assert sent["assistant_id"] == "asst-1" and sent["system_prompt"] == "be brief"
    assert "thread_id" not in sent  # fresh thread each call: replay-safe
    assert sent["content"] == "hello"
    assert out.usage_metadata["total_tokens"] == 5


async def test_tool_call_roundtrip_and_transcript_rendering():
    log: list = []
    m = model([call("add", {"a": 1, "b": 2}), text("3")], log).bind_tools([add])
    first = await m.ainvoke([HumanMessage("1+2?")])
    assert first.tool_calls[0]["name"] == "add" and first.tool_calls[0]["args"] == {"a": 1, "b": 2}
    assert log[0]["tools"][0]["function"]["name"] == "add"
    history = [HumanMessage("1+2?"), first, ToolMessage("3", tool_call_id="call_1", name="add")]
    await m.ainvoke(history)
    body = log[1]["content"]
    assert "tool_call id=call_1 name=add" in body and "tool_result id=call_1" in body


async def test_malformed_tool_arguments_become_invalid_tool_calls():
    m = model([call("add", "{not json")]).bind_tools([add])
    out = await m.ainvoke([HumanMessage("x")])
    assert not out.tool_calls and out.invalid_tool_calls[0]["name"] == "add"


def test_sync_invoke_works_inside_running_loop_and_outside():
    assert model([text("sync")]).invoke([HumanMessage("x")]).content == "sync"


async def test_sync_invoke_inside_running_loop():
    assert model([text("sync")]).invoke([HumanMessage("x")]).content == "sync"


async def test_empty_response_raises():
    from posttrain_agent.backboard_model import EmptyResponseError

    with pytest.raises(EmptyResponseError):
        await model([text("")]).ainvoke([HumanMessage("x")])


async def test_cassette_records_then_replays_without_client(tmp_path):
    path = tmp_path / "c.jsonl"
    live = model([text("recorded")], cassette=Cassette(path, "record"))
    assert (await live.ainvoke([HumanMessage("q")])).content == "recorded"

    def boom():
        raise AssertionError("client must not be used on replay")

    replay = ChatBackboard(client_factory=boom, cassette=Cassette(path, "replay"))
    assert (await replay.ainvoke([HumanMessage("q")])).content == "recorded"
    with pytest.raises(CassetteMiss):
        await replay.ainvoke([HumanMessage("different")])


async def test_server_error_propagates():
    with pytest.raises(BackboardServerError):
        await model([BackboardServerError("boom", 500)]).ainvoke([HumanMessage("x")])


def test_ai_message_with_text_and_calls_renders():
    from posttrain_agent.backboard_model import render_messages

    ai = AIMessage("thinking", tool_calls=[{"name": "add", "args": {"a": 1}, "id": "c1"}])
    _, body = render_messages([HumanMessage("q"), ai])
    assert "thinking" in body and "c1" in body
