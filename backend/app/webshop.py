"""Async client and deterministic repair verifier for AgentGym WebShop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.models import ToolCall


class WebShopProtocolError(RuntimeError):
    """Raised when the environment server violates its documented contract."""


@dataclass(frozen=True, slots=True)
class WebShopStep:
    state: str
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RepairVerification:
    original_reward: float
    candidate_reward: float
    verified: bool


def tool_call_to_action(call: ToolCall) -> str:
    """Convert the FunctionGemma tool contract into AgentGym action syntax."""

    if call.name == "search":
        value = call.arguments.get("keywords")
    elif call.name == "click":
        value = call.arguments.get("item")
    else:
        raise ValueError(f"unsupported WebShop tool {call.name!r}")
    if not isinstance(value, str) or not value.strip() or "]" in value:
        raise ValueError("WebShop action arguments must be non-empty strings without ']'")
    return f"{call.name}[{value.strip()}]"


class AgentGymWebShopClient:
    """Client for the HTTP API shipped by AgentGym's WebShop environment."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )
        self.env_id: int | None = None

    async def __aenter__(self) -> AgentGymWebShopClient:
        await self.create()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def create(self) -> int:
        response = await self._client.post("/create")
        response.raise_for_status()
        env_id = response.json()
        if not isinstance(env_id, int):
            raise WebShopProtocolError("/create must return an integer environment ID")
        self.env_id = env_id
        return env_id

    def _require_env(self) -> int:
        if self.env_id is None:
            raise WebShopProtocolError("create() must be called before using the environment")
        return self.env_id

    async def reset(self, session_id: int) -> str:
        response = await self._client.post(
            "/reset",
            json={"env_idx": self._require_env(), "session_id": session_id},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload or not isinstance(payload[0], str):
            raise WebShopProtocolError("/reset returned an invalid response")
        return await self.observation()

    async def observation(self) -> str:
        response = await self._client.get(
            "/observation",
            params={"env_idx": self._require_env()},
        )
        response.raise_for_status()
        observation = response.json()
        if not isinstance(observation, str):
            raise WebShopProtocolError("/observation must return a string")
        return observation

    async def available_actions(self) -> dict[str, Any]:
        response = await self._client.get(
            "/available_actions",
            params={"env_idx": self._require_env()},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise WebShopProtocolError("/available_actions must return an object")
        return payload

    async def step(self, action: ToolCall | str) -> WebShopStep:
        action_text = tool_call_to_action(action) if isinstance(action, ToolCall) else action
        response = await self._client.post(
            "/step",
            json={"env_idx": self._require_env(), "action": action_text},
        )
        response.raise_for_status()
        payload = response.json()
        try:
            return WebShopStep(
                state=str(payload["state"]),
                reward=float(payload["reward"]),
                done=bool(payload["done"]),
                info=dict(payload.get("info") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WebShopProtocolError("/step returned an invalid response") from exc

    async def rollout(self, session_id: int, actions: list[ToolCall | str]) -> float:
        await self.reset(session_id)
        reward = 0.0
        for action in actions:
            result = await self.step(action)
            reward = result.reward
            if result.done:
                break
        return reward

    async def verify_repair(
        self,
        *,
        session_id: int,
        prefix: list[ToolCall | str],
        original_continuation: list[ToolCall | str],
        candidate_continuation: list[ToolCall | str],
    ) -> RepairVerification:
        """Replay both complete continuations and admit only objective improvement."""

        original_reward = await self.rollout(session_id, [*prefix, *original_continuation])
        candidate_reward = await self.rollout(session_id, [*prefix, *candidate_continuation])
        return RepairVerification(
            original_reward=original_reward,
            candidate_reward=candidate_reward,
            verified=candidate_reward > original_reward,
        )

    async def close(self) -> None:
        if self.env_id is not None:
            response = await self._client.post("/close", json={"env_idx": self.env_id})
            response.raise_for_status()
            self.env_id = None
        await self._client.aclose()
