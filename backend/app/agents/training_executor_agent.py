"""Training execution role for adapter-backed post-training jobs.

This module is intentionally a contract adapter. It never invents provider job
ids, status transitions, artifact locations, metrics, or resource estimates.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from strands import Agent, tool

from .prompt_contract import get_prompt_contract, resolve_agent_model

_EVIDENCE_CLASSES = {"LIVE", "PRIOR_VERIFIED_RUN", "EXPLANATION"}
_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "STOPPED", "CANCELLED"}


def _stable_id(prefix: str, value: Any) -> str:
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


def _parse_object(value: Any, name: str) -> tuple[dict[str, Any] | None, str | None]:
    parsed = _as_json(value)
    if not isinstance(parsed, Mapping):
        return None, f"{name} must be a JSON object"
    return dict(parsed), None


class TrainingExecutorAgent:
    """Training Executor Agent; provider work is delegated to an adapter."""

    def __init__(
        self, model: str | None = None, adapter: Any = None, model_provider: Any = None
    ):
        self.adapter = adapter
        prompt_contract = get_prompt_contract("TrainingExecutorAgent")
        self.agent = Agent(
            name="TrainingExecutorAgent",
            model=resolve_agent_model(model, model_provider=model_provider),
            system_prompt=prompt_contract.prompt,
        )
        self.prompt_contract = prompt_contract
        self.prompt_metadata = prompt_contract.metadata()
        self.agent.tool_registry.register_tool(self.submit_training_job)
        self.agent.tool_registry.register_tool(self.monitor_training_job)
        self.agent.tool_registry.register_tool(self.handle_training_failure)
        self.agent.tool_registry.register_tool(self.retrieve_trained_artifact)

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
        except Exception as exc:
            return {
                "status": "adapter_error",
                "adapter_error": type(exc).__name__,
                "evidence_class": "EXPLANATION",
            }
        parsed = _as_json(raw)
        result = dict(parsed) if isinstance(parsed, Mapping) else {"adapter_result": parsed}
        result.setdefault("evidence_class", _evidence_class(result))
        return result

    @tool
    def submit_training_job(
        self,
        configuration: str,
        dataset_references: str,
        base_model: str,
        job_name: str | None = None,
    ) -> str:
        """Submit a validated training request through the provider adapter."""
        config, config_error = _parse_object(configuration, "configuration")
        data_refs, refs_error = _parse_object(dataset_references, "dataset_references")
        if config_error or refs_error:
            return json.dumps(
                {
                    "status": "invalid_request",
                    "error": config_error or refs_error,
                    "evidence_class": "EXPLANATION",
                },
                indent=2,
            )

        request = {
            "configuration": config,
            "dataset_references": data_refs,
            "base_model": base_model,
            "job_name": job_name,
        }
        result = self._call_adapter("submit_training_job", **request)
        if result is None:
            result = {
                "status": "adapter_required",
                "job_id": job_name,
                "submission_accepted": False,
                "adapter_required_for": "training job submission",
                "evidence_class": "EXPLANATION",
            }
        result.setdefault("request_id", _stable_id("training_request", request))
        result.setdefault("configuration_used", config)
        result.setdefault("dataset_references", data_refs)
        result.setdefault("base_model", base_model)
        result.setdefault("evidence_class", _evidence_class(result))
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    @tool
    def monitor_training_job(self, job_id: str, poll_interval_seconds: int = 30) -> str:
        """Read provider status once; polling is controlled by the caller."""
        if (
            not isinstance(poll_interval_seconds, int)
            or isinstance(poll_interval_seconds, bool)
            or poll_interval_seconds < 0
        ):
            return json.dumps(
                {
                    "status": "invalid_request",
                    "error": "poll_interval_seconds must be a non-negative integer",
                    "evidence_class": "EXPLANATION",
                },
                indent=2,
            )
        request = {"job_id": job_id, "poll_interval_seconds": poll_interval_seconds}
        result = self._call_adapter("monitor_training_job", **request)
        if result is None:
            result = {
                "status": "adapter_required",
                "job_id": job_id,
                "job_status": None,
                "adapter_required_for": "training provider status polling",
                "evidence_class": "EXPLANATION",
            }
        result.setdefault("request_id", _stable_id("training_status", request))
        result.setdefault("job_id", job_id)
        result.setdefault("evidence_class", _evidence_class(result))
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    @tool
    def handle_training_failure(self, job_status: str, failure_info: str | None = None) -> str:
        """Describe deterministic next steps without claiming actions occurred."""
        parsed_info = _as_json(failure_info) if failure_info else {}
        if not isinstance(parsed_info, Mapping):
            parsed_info = {"raw_info": parsed_info}
        adapter_result = self._call_adapter(
            "handle_training_failure", job_status=job_status, failure_info=dict(parsed_info)
        )
        if adapter_result is not None:
            result = adapter_result
        else:
            known = job_status in _TERMINAL_STATUSES or job_status == "IN_PROGRESS"
            result = {
                "status": "handled_locally",
                "job_status": job_status,
                "is_terminal": job_status in _TERMINAL_STATUSES,
                "was_successful": job_status == "COMPLETED",
                "actions_taken": [],
                "recommendations": (
                    ["Inspect provider failure details before retrying"]
                    if job_status in {"FAILED", "STOPPED", "CANCELLED"}
                    else ["Continue provider polling"]
                    if job_status == "IN_PROGRESS"
                    else []
                ),
                "next_steps": (
                    ["Retry only after the provider and input state are validated"]
                    if job_status in {"FAILED", "STOPPED", "CANCELLED"}
                    else ["Wait for the next status response"]
                    if job_status == "IN_PROGRESS"
                    else []
                ),
                "failure_info": dict(parsed_info),
                "status_known": known,
                "evidence_class": _evidence_class(parsed_info),
            }
        result.setdefault(
            "handling_id",
            _stable_id("failure_handling", {"job_status": job_status, "failure_info": parsed_info}),
        )
        result.setdefault("job_status", job_status)
        result.setdefault("evidence_class", _evidence_class(result))
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    @tool
    def retrieve_trained_artifact(self, job_status: str, job_info: str) -> str:
        """Retrieve provider-owned artifacts without synthesizing locations."""
        info, info_error = _parse_object(job_info, "job_info")
        if info_error:
            return json.dumps(
                {"status": "invalid_request", "error": info_error, "evidence_class": "EXPLANATION"},
                indent=2,
            )
        observed_status = info.get("job_status", job_status)
        if observed_status != "COMPLETED":
            return json.dumps(
                {
                    "status": "not_ready",
                    "job_status": observed_status,
                    "artifact_status": "unavailable",
                    "error": "artifacts require a completed provider job",
                    "evidence_class": _evidence_class(info),
                },
                indent=2,
                sort_keys=True,
            )

        request = {"job_status": job_status, "job_info": info}
        result = self._call_adapter("retrieve_trained_artifact", **request)
        if result is None:
            result = {
                "status": "adapter_required",
                "artifact_status": "unavailable",
                "job_id": info.get("job_id"),
                "adapter_required_for": "trained artifact retrieval and validation",
                "evidence_class": "EXPLANATION",
            }
        result.setdefault("retrieval_id", _stable_id("artifact_retrieval", request))
        result.setdefault("job_id", info.get("job_id"))
        result.setdefault("evidence_class", _evidence_class(result))
        return json.dumps(result, indent=2, sort_keys=True, default=str)


def create_training_executor_agent(
    model: str | None = None, adapter: Any = None, model_provider: Any = None
) -> TrainingExecutorAgent:
    """Create a Training Executor Agent with an optional provider adapter."""
    return TrainingExecutorAgent(model, adapter, model_provider)
