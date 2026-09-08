"""Evaluation role for adapter-backed, reproducible measurements.

Evaluation metrics are evidence, not estimates. The evaluator delegates model
execution to an adapter and performs only deterministic calculations over
returned measurements.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from strands import Agent, tool

from .prompt_contract import get_prompt_contract, resolve_nemotron_model

_EVIDENCE_CLASSES = {"LIVE", "PRIOR_VERIFIED_RUN", "EXPLANATION"}


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


def _combine_evidence(*values: Any) -> str:
    classes = {_evidence_class(value) for value in values}
    if len(classes) == 1:
        return classes.pop()
    return "EXPLANATION"


def _object(value: Any, name: str) -> tuple[dict[str, Any] | None, str | None]:
    parsed = _as_json(value)
    if not isinstance(parsed, Mapping):
        return None, f"{name} must be a JSON object"
    return dict(parsed), None


class EvalAgent:
    """Eval Agent; model execution is delegated to an injected adapter."""

    def __init__(self, model: str | None = None, adapter: Any = None):
        self.adapter = adapter
        prompt_contract = get_prompt_contract("EvalAgent")
        self.agent = Agent(
            name="EvalAgent",
            model=resolve_nemotron_model(model),
            system_prompt=prompt_contract.prompt,
        )
        self.prompt_contract = prompt_contract
        self.prompt_metadata = prompt_contract.metadata()
        self.agent.tool_registry.register_tool(self.evaluate_held_out_performance)
        self.agent.tool_registry.register_tool(self.run_regression_benchmarks)
        self.agent.tool_registry.register_tool(self.calculate_performance_metrics)
        self.agent.tool_registry.register_tool(self.assess_statistical_significance)

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
    def evaluate_held_out_performance(
        self,
        model_artifacts: str,
        evaluation_config: dict[str, Any],
        num_episodes: int = 20,
    ) -> str:
        """Request held-out evaluation; no local score is inferred."""
        artifacts, artifacts_error = _object(model_artifacts, "model_artifacts")
        if artifacts_error:
            return json.dumps(
                {
                    "status": "invalid_request",
                    "error": artifacts_error,
                    "evidence_class": "EXPLANATION",
                },
                indent=2,
            )
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
            "model_artifacts": artifacts,
            "evaluation_config": evaluation_config,
            "num_episodes": num_episodes,
        }
        result = self._call_adapter("evaluate_held_out_performance", **request)
        if result is None:
            result = {
                "status": "adapter_required",
                "evaluation_type": "held_out_performance",
                "metrics_available": False,
                "adapter_required_for": "held-out model execution",
                "evidence_class": "EXPLANATION",
            }
        result.setdefault("evaluation_id", _stable_id("heldout_evaluation", request))
        result.setdefault("evaluation_type", "held_out_performance")
        result.setdefault("evaluation_config", evaluation_config)
        result.setdefault("num_episodes", num_episodes)
        result.setdefault("evidence_class", _evidence_class(result))
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    @tool
    def run_regression_benchmarks(
        self,
        model_artifacts: str,
        regression_suite: dict[str, Any],
        baseline_performance: float | None = None,
    ) -> str:
        """Request regression evaluation; baseline values are passed through."""
        artifacts, artifacts_error = _object(model_artifacts, "model_artifacts")
        if artifacts_error:
            return json.dumps(
                {
                    "status": "invalid_request",
                    "error": artifacts_error,
                    "evidence_class": "EXPLANATION",
                },
                indent=2,
            )
        if baseline_performance is not None and (
            not isinstance(baseline_performance, (int, float))
            or isinstance(baseline_performance, bool)
        ):
            return json.dumps(
                {
                    "status": "invalid_request",
                    "error": "baseline_performance must be numeric",
                    "evidence_class": "EXPLANATION",
                },
                indent=2,
            )
        request = {
            "model_artifacts": artifacts,
            "regression_suite": regression_suite,
            "baseline_performance": baseline_performance,
        }
        result = self._call_adapter("run_regression_benchmarks", **request)
        if result is None:
            result = {
                "status": "adapter_required",
                "evaluation_type": "regression_benchmark",
                "metrics_available": False,
                "adapter_required_for": "regression model execution",
                "evidence_class": "EXPLANATION",
            }
        result.setdefault("evaluation_id", _stable_id("regression_evaluation", request))
        result.setdefault("evaluation_type", "regression_benchmark")
        result.setdefault("regression_suite", regression_suite)
        result.setdefault("baseline_performance_reference", baseline_performance)
        result.setdefault("evidence_class", _evidence_class(result))
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    @tool
    def calculate_performance_metrics(self, held_out_results: str, regression_results: str) -> str:
        """Combine concrete evaluator measurements deterministically."""
        held_out, held_out_error = _object(held_out_results, "held_out_results")
        regression, regression_error = _object(regression_results, "regression_results")
        if held_out_error or regression_error:
            return json.dumps(
                {
                    "status": "invalid_request",
                    "error": held_out_error or regression_error,
                    "evidence_class": "EXPLANATION",
                },
                indent=2,
            )

        held_out_metrics = held_out.get("performance_metrics")
        if not isinstance(held_out_metrics, Mapping):
            held_out_metrics = {}
        held_out_score = held_out_metrics.get("success_rate")
        if held_out_score is None:
            combined = held_out.get("combined_performance_metrics")
            if isinstance(combined, Mapping):
                held_out_score = combined.get("held_out_success_rate")

        regression_summary = regression.get("regression_summary")
        if not isinstance(regression_summary, Mapping):
            regression_summary = {}
        regression_penalty = regression.get(
            "overall_regression_score", regression_summary.get("overall_regression_score")
        )

        if (
            not isinstance(held_out_score, (int, float))
            or isinstance(held_out_score, bool)
            or not isinstance(regression_penalty, (int, float))
            or isinstance(regression_penalty, bool)
        ):
            return json.dumps(
                {
                    "status": "insufficient_evidence",
                    "metrics_calculated": False,
                    "missing_measurements": [
                        name
                        for name, value in (
                            ("held_out_success_rate", held_out_score),
                            ("overall_regression_score", regression_penalty),
                        )
                        if not isinstance(value, (int, float)) or isinstance(value, bool)
                    ],
                    "evidence_class": _combine_evidence(held_out, regression),
                },
                indent=2,
            )

        combined_score = max(
            0.0, min(1.0, float(held_out_score) - (float(regression_penalty) * 0.5))
        )
        result = {
            "status": "completed",
            "metrics_calculation_id": _stable_id(
                "metrics", {"held_out": held_out, "regression": regression}
            ),
            "metrics_calculated": True,
            "evidence_class": _combine_evidence(held_out, regression),
            "held_out_evaluation": {
                "success_rate": held_out_score,
                "mean_episode_reward": held_out_metrics.get("mean_episode_reward"),
                "evaluation_id": held_out.get("evaluation_id"),
            },
            "regression_evaluation": {
                "overall_regression_score": regression_penalty,
                "regression_detected": regression_summary.get("overall_regression_detected"),
                "evaluation_id": regression.get("evaluation_id"),
            },
            "combined_performance_metrics": {
                "held_out_success_rate": held_out_score,
                "regression_penalty": round(float(regression_penalty), 6),
                "combined_score": round(combined_score, 6),
            },
        }
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    @tool
    def assess_statistical_significance(
        self, eval_results: str, baseline_metrics: dict[str, Any] | None = None
    ) -> str:
        """Apply a deterministic two-proportion test to supplied measurements."""
        results, results_error = _object(eval_results, "eval_results")
        if results_error:
            return json.dumps(
                {
                    "status": "invalid_request",
                    "error": results_error,
                    "evidence_class": "EXPLANATION",
                },
                indent=2,
            )
        baseline = baseline_metrics if isinstance(baseline_metrics, Mapping) else {}
        metrics = results.get("performance_metrics")
        if not isinstance(metrics, Mapping):
            metrics = results.get("combined_performance_metrics")
        if not isinstance(metrics, Mapping):
            metrics = {}
        eval_rate = metrics.get("success_rate", metrics.get("held_out_success_rate"))
        eval_n = results.get("num_episodes")
        if not isinstance(eval_n, int) or isinstance(eval_n, bool):
            raw = results.get("raw_results")
            eval_n = raw.get("total_episodes") if isinstance(raw, Mapping) else None
        baseline_rate = baseline.get("success_rate")
        baseline_n = baseline.get("sample_size")
        missing = []
        if not isinstance(eval_rate, (int, float)) or isinstance(eval_rate, bool):
            missing.append("evaluation success_rate")
        if not isinstance(eval_n, int) or isinstance(eval_n, bool) or eval_n <= 0:
            missing.append("evaluation sample_size")
        if not isinstance(baseline_rate, (int, float)) or isinstance(baseline_rate, bool):
            missing.append("baseline success_rate")
        if not isinstance(baseline_n, int) or isinstance(baseline_n, bool) or baseline_n <= 0:
            missing.append("baseline sample_size")
        if missing:
            return json.dumps(
                {
                    "status": "insufficient_evidence",
                    "significance_assessed": False,
                    "missing_measurements": missing,
                    "evidence_class": _combine_evidence(results, baseline),
                },
                indent=2,
            )

        pooled = (float(eval_rate) * eval_n + float(baseline_rate) * baseline_n) / (
            eval_n + baseline_n
        )
        variance = pooled * (1 - pooled) * (1 / eval_n + 1 / baseline_n)
        se = math.sqrt(max(0.0, variance))
        z_score = (float(eval_rate) - float(baseline_rate)) / se if se else 0.0
        p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
        effect_size = 2 * (
            math.asin(math.sqrt(float(eval_rate))) - math.asin(math.sqrt(float(baseline_rate)))
        )
        significant = p_value < 0.05
        result = {
            "status": "completed",
            "significance_assessment_id": _stable_id(
                "significance", {"results": results, "baseline": baseline}
            ),
            "evidence_class": _combine_evidence(results, baseline),
            "evaluation_metadata": {
                "eval_success_rate": eval_rate,
                "eval_sample_size": eval_n,
                "baseline_success_rate": baseline_rate,
                "baseline_sample_size": baseline_n,
            },
            "statistical_tests": {
                "test_type": "two_proportion_z_test",
                "z_score": round(z_score, 6),
                "p_value": round(p_value, 6),
                "is_statistically_significant": significant,
                "alpha_level": 0.05,
            },
            "effect_size": {
                "effect_size_measure": "cohens_h",
                "effect_size_value": round(effect_size, 6),
            },
            "significance_summary": {
                "significance_assessment_completed": True,
                "performance_improvement_statistically_significant": significant
                and eval_rate > baseline_rate,
                "performance_degradation_statistically_significant": significant
                and eval_rate < baseline_rate,
                "ready_for_decision_making": True,
            },
        }
        return json.dumps(result, indent=2, sort_keys=True, default=str)


def create_eval_agent(model: str | None = None, adapter: Any = None) -> EvalAgent:
    """Create an Eval Agent with an optional objective adapter."""
    return EvalAgent(model, adapter)
