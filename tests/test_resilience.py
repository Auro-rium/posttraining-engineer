import pytest
from backboard import (
    BackboardAPIError,
    BackboardRateLimitError,
    BackboardServerError,
    BackboardValidationError,
)
from fakes import FakeBackboard, text
from langchain.agents import create_agent

from posttrain_agent.config import Settings
from posttrain_agent.resilience import build_middleware, build_model, is_transient


def test_transient_classification():
    assert is_transient(BackboardServerError("x", 503))
    assert is_transient(BackboardRateLimitError("x", 429))
    assert is_transient(BackboardAPIError("Request timed out"))  # SDK: timeout has no status
    assert not is_transient(BackboardAPIError("nope", 401))
    assert not is_transient(BackboardValidationError("bad", 400))
    assert not is_transient(ValueError("x"))


def _agent(settings, script, log):
    factory = lambda: FakeBackboard(script, log)
    return create_agent(
        build_model(settings, client_factory=factory),
        tools=[],
        middleware=build_middleware(settings, client_factory=factory),
    )


async def test_retries_transient_error_on_same_model(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr("asyncio.sleep", _instant)
    log: list = []
    seq = [BackboardServerError("blip", 500), text("ok")]
    agent = _agent(Settings(model_retries=2), lambda kw: seq.pop(0), log)
    out = await agent.ainvoke({"messages": [("user", "hi")]})
    assert out["messages"][-1].content == "ok"
    assert [k["model_name"] for k in log] == [Settings().model_name] * 2


async def test_falls_back_to_second_model_after_retries(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _instant)
    log: list = []

    def script(kw):
        if kw["model_name"] == "primary":
            return BackboardServerError("down", 500)
        return text("from fallback")

    s = Settings(model_name="primary", llm_provider="p1", model_retries=1,
                 fallback_models=(("p2", "backup"),))
    out = await _agent(s, script, log).ainvoke({"messages": [("user", "hi")]})
    assert out["messages"][-1].content == "from fallback"
    names = [k["model_name"] for k in log]
    assert names == ["primary", "primary", "backup"]  # 1 try + 1 retry, then fallback


async def test_non_transient_error_is_not_retried_but_still_falls_back(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _instant)
    log: list = []

    def script(kw):
        return BackboardValidationError("bad request", 400) if kw["model_name"] == "primary" \
            else text("fb")

    s = Settings(model_name="primary", fallback_models=(("p2", "backup"),))
    out = await _agent(s, script, log).ainvoke({"messages": [("user", "hi")]})
    assert out["messages"][-1].content == "fb"
    assert [k["model_name"] for k in log] == ["primary", "backup"]


async def test_all_models_failing_raises(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _instant)
    s = Settings(model_retries=0, fallback_models=(("p2", "backup"),))
    with pytest.raises(BackboardServerError):
        await _agent(s, lambda kw: BackboardServerError("x", 500), []).ainvoke(
            {"messages": [("user", "hi")]}
        )


async def _instant(_s):
    return None
