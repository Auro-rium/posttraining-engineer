"""Data-curation role for adapter-backed post-training pipelines.

The curator may format records deterministically once verification evidence is
present. Decision discovery, correction generation, verification, and artifact
storage remain adapter operations so an explanatory run cannot become training
data by accident.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from strands import Agent, tool

from .prompt_contract import get_prompt_contract, resolve_nemotron_model

_EVIDENCE_CLASSES = {"LIVE", "PRIOR_VERIFIED_RUN", "EXPLANATION"}
_VERIFIED_EVIDENCE = {"LIVE", "PRIOR_VERIFIED_RUN"}


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


class DataCuratorAgent:
    """Data Curator Agent; authoritative operations are adapter-owned."""

    def __init__(self, model: str | None = None, adapter: Any = None):
        self.adapter = adapter
        prompt_contract = get_prompt_contract("DataCuratorAgent")
        self.agent = Agent(
            name="DataCuratorAgent",
            model=resolve_nemotron_model(model),
            system_prompt=prompt_contract.prompt,
        )
        self.prompt_contract = prompt_contract
        self.prompt_metadata = prompt_contract.metadata()
        self.agent.tool_registry.register_tool(self.identify_decision_points)
        self.agent.tool_registry.register_tool(self.generate_corrected_trajectories)
        self.agent.tool_registry.register_tool(self.verify_corrections)
        self.agent.tool_registry.register_tool(self.format_sft_dataset)

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
    def identify_decision_points(self, trajectories: str, failure_analysis: str) -> str:
        """Ask the objective adapter to identify evidence-backed decision points."""
        traj_data = _as_json(trajectories)
        fail_data = _as_json(failure_analysis)
        if not isinstance(traj_data, Mapping) or not isinstance(fail_data, Mapping):
            return json.dumps(
                {
                    "status": "invalid_request",
                    "error": "trajectories and failure_analysis must be JSON objects",
                    "evidence_class": "EXPLANATION",
                },
                indent=2,
            )
        request = {"trajectories": dict(traj_data), "failure_analysis": dict(fail_data)}
        result = self._call_adapter("identify_decision_points", **request)
        if result is None:
            result = {
                "status": "adapter_required",
                "decision_points_identified": 0,
                "decision_points": [],
                "adapter_required_for": "trajectory inspection and decision-point identification",
                "evidence_class": "EXPLANATION",
            }
        result.setdefault("curation_id", _stable_id("curation", request))
        result.setdefault("decision_points", [])
        result.setdefault(
            "decision_points_identified",
            len(result["decision_points"]) if isinstance(result["decision_points"], list) else 0,
        )
        result.setdefault("evidence_class", _evidence_class(result))
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    @tool
    def generate_corrected_trajectories(
        self, decision_points: str, num_variations_per_point: int = 3
    ) -> str:
        """Ask the objective adapter to generate candidate corrections."""
        points_data = _as_json(decision_points)
        if not isinstance(points_data, Mapping):
            return json.dumps(
                {
                    "status": "invalid_request",
                    "error": "decision_points must be a JSON object",
                    "evidence_class": "EXPLANATION",
                },
                indent=2,
            )
        if (
            not isinstance(num_variations_per_point, int)
            or isinstance(num_variations_per_point, bool)
            or num_variations_per_point < 0
        ):
            return json.dumps(
                {
                    "status": "invalid_request",
                    "error": "num_variations_per_point must be a non-negative integer",
                    "evidence_class": "EXPLANATION",
                },
                indent=2,
            )
        request = {
            "decision_points": dict(points_data),
            "num_variations_per_point": num_variations_per_point,
        }
        result = self._call_adapter("generate_corrected_trajectories", **request)
        if result is None:
            result = {
                "status": "adapter_required",
                "corrected_trajectories": [],
                "total_trajectories_generated": 0,
                "adapter_required_for": "correction generation",
                "evidence_class": "EXPLANATION",
            }
        result.setdefault("trajectory_generation_id", _stable_id("trajectory_generation", request))
        result.setdefault("corrected_trajectories", [])
        result.setdefault("evidence_class", _evidence_class(result))
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    @tool
    def verify_corrections(
        self, corrected_trajectories: str, environment_config: dict[str, Any]
    ) -> str:
        """Verify corrections through the objective adapter; fail closed locally."""
        traj_data = _as_json(corrected_trajectories)
        if not isinstance(traj_data, Mapping):
            return json.dumps(
                {
                    "status": "invalid_request",
                    "error": "corrected_trajectories must be a JSON object",
                    "evidence_class": "EXPLANATION",
                },
                indent=2,
            )
        request = {
            "corrected_trajectories": dict(traj_data),
            "environment_config": environment_config,
        }
        result = self._call_adapter("verify_corrections", **request)
        if result is None:
            submitted = traj_data.get("corrected_trajectories", [])
            result = {
                "status": "adapter_required",
                "trajectories_submitted": len(submitted) if isinstance(submitted, list) else 0,
                "verified_trajectories": [],
                "failed_trajectories": [],
                "adapter_required_for": "environment replay and correction verification",
                "evidence_class": "EXPLANATION",
            }
        result.setdefault("verification_id", _stable_id("verification", request))
        result.setdefault("verified_trajectories", [])
        result.setdefault("evidence_class", _evidence_class(result))
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    @tool
    def format_sft_dataset(
        self, verified_trajectories: str, format_type: str = "conversational"
    ) -> str:
        """Format only adapter-verified records; storage is adapter-owned."""
        data = _as_json(verified_trajectories)
        if not isinstance(data, Mapping):
            return json.dumps(
                {
                    "status": "invalid_request",
                    "error": "verified_trajectories must be a JSON object",
                    "evidence_class": "EXPLANATION",
                },
                indent=2,
            )

        adapter_result = self._call_adapter(
            "format_sft_dataset", verified_trajectories=dict(data), format_type=format_type
        )
        if adapter_result is not None:
            result = adapter_result
            result.setdefault(
                "dataset_id", _stable_id("sft_dataset", {"data": data, "format_type": format_type})
            )
            result.setdefault("evidence_class", _evidence_class(result))
            return json.dumps(result, indent=2, sort_keys=True, default=str)

        records = data.get("verified_trajectories", [])
        evidence = _evidence_class(data)
        eligible = records if evidence in _VERIFIED_EVIDENCE and isinstance(records, list) else []
        eligible = [
            record
            for record in eligible
            if isinstance(record, Mapping)
            and record.get("verification_status") == "verified"
            and (
                _evidence_class(record)
                if "evidence_class" in record
                else evidence
            )
            in _VERIFIED_EVIDENCE
        ]
        examples = []
        for record in eligible:
            teaching = record.get("teaching_example", {})
            if not isinstance(teaching, Mapping):
                continue
            situation = teaching.get("situation")
            correct_behavior = teaching.get("correct_behavior")
            if not isinstance(situation, str) or not isinstance(correct_behavior, str):
                continue
            examples.append(
                {
                    "example_id": _stable_id("sft_example", record),
                    "source_trajectory": record.get("trajectory_id"),
                    "conversation": [
                        {
                            "role": "system",
                            "content": (
                                "Always verify that a service repair works before "
                                "declaring success."
                            ),
                        },
                        {"role": "user", "content": f"Service issue context: {situation}"},
                        {"role": "assistant", "content": correct_behavior},
                    ],
                    "metadata": {
                        "verified_in_environment": True,
                        "verification_timestamp": record.get("verification_timestamp"),
                    },
                }
            )

        request = {"verified_trajectories": data, "format_type": format_type}
        result = {
            "status": "formatted" if examples else "insufficient_evidence",
            "dataset_id": _stable_id("sft_dataset", request),
            "format_type": format_type,
            "verified_trajectories_used": len(eligible),
            "sft_examples_created": len(examples),
            "sft_examples": examples,
            "dataset_references": [],
            "storage_status": "adapter_required",
            "evidence_class": evidence,
            "creation_summary": {
                "data_source": "environment_verified_corrections",
                "no_unverified_data_included": True,
                "ready_for_training": bool(examples),
            },
        }
        return json.dumps(result, indent=2, sort_keys=True, default=str)


def create_data_curator_agent(model: str | None = None, adapter: Any = None) -> DataCuratorAgent:
    """Create a Data Curator Agent with an optional objective adapter."""
    return DataCuratorAgent(model, adapter)
