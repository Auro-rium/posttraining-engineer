"""Benchmark role for adapter-backed post-training measurements.

The role owns the contract for benchmark requests, but it does not invent
trajectories, metrics, or object-store locations. A configured adapter must
perform the actual benchmark and return its evidence classification.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from strands import Agent, tool

from .prompt_contract import get_prompt_contract, resolve_agent_model

_EVIDENCE_CLASSES = {"LIVE", "PRIOR_VERIFIED_RUN", "EXPLANATION"}


def _stable_id(prefix: str, value: Any) -> str:
    """Return an id stable for the same request payload."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha256(encoded.encode()).hexdigest()[:16]}"


def _as_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _evidence_class(value: Any) -> str:
    if isinstance(value, Mapping) and value.get("evidence_class") in _EVIDENCE_CLASSES:
        return str(value["evidence_class"])
    return "EXPLANATION"


class BenchmarkAgent:
    """Benchmark Agent; execution is delegated to an injected adapter."""

    def __init__(
        self, model: str | None = None, adapter: Any = None, model_provider: Any = None
    ):
        self.adapter = adapter
        prompt_contract = get_prompt_contract("BenchmarkAgent")
        self.agent = Agent(
            name="BenchmarkAgent",
            model=resolve_agent_model(model, model_provider=model_provider),
            system_prompt=prompt_contract.prompt,
        )
        self.prompt_contract = prompt_contract
        self.prompt_metadata = prompt_contract.metadata()
        self.agent.tool_registry.register_tool(self.execute_benchmark)
        self.agent.tool_registry.register_tool(self.record_trajectory)
        self.agent.tool_registry.register_tool(self.calculate_metrics)

    def _call_adapter(self, operation: str, **kwargs: Any) -> dict[str, Any] | None:
        if self.adapter is None:
            return None
        target = getattr(self.adapter, operation, None)
        if target is None and callable(self.adapter):
            target = self.adapter
        if target is None:
            return {
                "status": "adapter_error",
                "adapter_error": f"adapter does not implement {operation}",
                "evidence_class": "EXPLANATION",
            }
        try:
            try:
                raw = target(**kwargs)
            except TypeError:
                raw = target(kwargs)
        except Exception as exc:  # adapter boundary: never claim a measurement
            return {
                "status": "adapter_error",
                "adapter_error": type(exc).__name__,
                "evidence_class": "EXPLANATION",
            }
        parsed = _as_json(raw)
        if isinstance(parsed, Mapping):
            result = dict(parsed)
        else:
            result = {"adapter_result": parsed}
        result.setdefault("evidence_class", _evidence_class(result))
        return result

    @tool
    def execute_benchmark(
        self,
        run_id: str,
        environment_config: dict[str, Any],
        num_episodes: int = 10,
    ) -> str:
        """Request benchmark execution and return adapter-owned evidence."""
        if not isinstance(num_episodes, int) or isinstance(num_episodes, bool) or num_episodes < 0:
            return json.dumps(
                {
                    "status": "invalid_request",
                    "error": "num_episodes must be a non-negative integer",
                    "evidence_class": "EXPLANATION",
                },
                indent=2,
            )

        request = {
            "run_id": run_id,
            "environment_config": environment_config,
            "num_episodes": num_episodes,
        }
        result = self._call_adapter("execute_benchmark", **request)
        if result is None:
            result = {
                "status": "adapter_required",
                "benchmark_completed": False,
                "trajectories_collected": 0,
                "trajectory_references": [],
                "metrics_available": False,
                "adapter_required_for": "objective benchmark execution",
                "evidence_class": "EXPLANATION",
            }
        result.setdefault("benchmark_id", _stable_id("benchmark", request))
        result.setdefault("run_id", run_id)
        result.setdefault("benchmark_type", "initial_evaluation")
        result.setdefault("num_episodes", num_episodes)
        result.setdefault("environment_info", environment_config)
        result.setdefault("trajectory_references", [])
        result.setdefault("evidence_class", _evidence_class(result))
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    @tool
    def record_trajectory(self, episode_data: dict[str, Any]) -> str:
        """Persist a trajectory through the configured artifact adapter."""
        request = {"episode_data": episode_data}
        result = self._call_adapter("record_trajectory", **request)
        if result is None:
            result = {
                "status": "adapter_required",
                "stored": False,
                "trajectory_reference": None,
                "adapter_required_for": "trajectory persistence",
                "evidence_class": "EXPLANATION",
            }
        result.setdefault("trajectory_id", _stable_id("trajectory", episode_data))
        result.setdefault("stored", result.get("status") == "completed")
        result.setdefault("evidence_class", _evidence_class(result))
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    @tool
    def calculate_metrics(self, trajectories: list[str]) -> str:
        """Calculate metrics through an adapter; references alone are not metrics."""
        request = {"trajectories": trajectories}
        result = self._call_adapter("calculate_metrics", **request)
        if result is None:
            result = {
                "status": "adapter_required",
                "num_trajectories": len(trajectories),
                "metrics_calculated": False,
                "adapter_required_for": "trajectory loading and metric calculation",
                "evidence_class": "EXPLANATION",
            }
        result.setdefault("metrics_id", _stable_id("metrics", request))
        result.setdefault("num_trajectories", len(trajectories))
        result.setdefault("evidence_class", _evidence_class(result))
        return json.dumps(result, indent=2, sort_keys=True, default=str)


def create_benchmark_agent(
    model: str | None = None, adapter: Any = None, model_provider: Any = None
) -> BenchmarkAgent:
    """Create a Benchmark Agent with an optional objective adapter."""
    return BenchmarkAgent(model, adapter, model_provider)
