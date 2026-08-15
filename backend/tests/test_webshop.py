from __future__ import annotations

import json

import httpx
import pytest

from app.models import ToolCall
from app.webshop import AgentGymWebShopClient, tool_call_to_action


def test_tool_call_to_action_uses_agentgym_function_contract() -> None:
    assert tool_call_to_action(ToolCall(name="search", arguments={"keywords": "blue shoes"})) == (
        "search[blue shoes]"
    )
    assert tool_call_to_action(ToolCall(name="click", arguments={"item": "Buy Now"})) == (
        "click[Buy Now]"
    )
    with pytest.raises(ValueError, match="unsupported"):
        tool_call_to_action(ToolCall(name="delete", arguments={}))


@pytest.mark.asyncio
async def test_repair_verifier_replays_both_complete_trajectories() -> None:
    actions: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/create":
            return httpx.Response(200, json=7)
        if request.url.path == "/reset":
            actions.clear()
            return httpx.Response(200, json=["reset", None])
        if request.url.path == "/observation":
            return httpx.Response(200, json="WebShop [SEP] Search")
        if request.url.path == "/step":
            payload = json.loads(request.content)
            actions.append(payload["action"])
            reward = 1.0 if actions[-1] == "click[Buy Now]" else 0.0
            return httpx.Response(
                200,
                json={"state": "done", "reward": reward, "done": reward == 1.0, "info": {}},
            )
        if request.url.path == "/close":
            return httpx.Response(200, json=None)
        raise AssertionError(request.url)

    client = AgentGymWebShopClient(
        "http://agentgym",
        transport=httpx.MockTransport(handler),
    )
    await client.create()
    result = await client.verify_repair(
        session_id=12,
        prefix=["search[navy shorts]"],
        original_continuation=["click[wrong product]"],
        candidate_continuation=["click[Buy Now]"],
    )
    assert result.verified is True
    assert result.original_reward == 0.0
    assert result.candidate_reward == 1.0
    await client.close()
