"""Deterministic post-training rules. Nothing here calls an LLM.

Agents may propose; only this code validates configurations, enforces the sealed
held-out boundary and decides promotion. Every measurement carries an evidence
label so simulated numbers can never be mistaken for measured ones.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from math import isfinite
from typing import Any, Literal, Protocol

Evidence = Literal["LIVE", "PRIOR_VERIFIED_RUN", "EXPLANATION"]
VERIFIED: frozenset[str] = frozenset({"LIVE", "PRIOR_VERIFIED_RUN"})
OBJECTIVE_TOOLS = ("get_logs", "inspect_service", "read_config", "edit_config",
                   "restart_service", "run_healthcheck")

QLORA_SPACE: dict[str, Any] = {
    "rank": (4, 8, 16, 32, 64),
    "alpha": (8, 16, 32, 64, 128),
    "dropout": (0.0, 0.1),  # inclusive range
    "learning_rate": (1e-5, 3e-4),  # inclusive range
    "epochs": (1, 5),
    "sequence_length": (512, 1024, 2048),
    "batch_size": (1, 2, 4, 8),
    "gradient_accumulation_steps": (1, 2, 4, 8, 16),
}
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

MIN_RELATIVE_IMPROVEMENT = 0.10
MAX_ABSOLUTE_REGRESSION = 0.05


class ExperimentBackend(Protocol):
    """Where benchmarks, training and evaluation actually run (SageMaker, local GPU, ...).

    Implementations must return provider-verified numbers and label them ``LIVE``.
    """

    def benchmark(self, checkpoint: str, suite: str, seed: int) -> dict[str, Any]: ...
    def train(self, config: dict[str, Any], dataset_ref: str, base_checkpoint: str) -> dict[str, Any]: ...
    def evaluate(self, checkpoint: str, suite: str, split: str, seed: int) -> dict[str, Any]: ...


def _unit(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class LocalStubBackend:
    """Deterministic simulation. Every result is labelled ``EXPLANATION``."""

    def benchmark(self, checkpoint: str, suite: str, seed: int) -> dict[str, Any]:
        score = round(0.30 + 0.30 * _unit(checkpoint, suite, "train", seed), 4)
        return {"evidence": "EXPLANATION", "checkpoint": checkpoint, "suite": suite,
                "split": "train", "seed": seed, "score": score,
                "trajectory_refs": [f"trajectory://stub/{checkpoint}/{i}" for i in range(3)]}

    def train(self, config: dict[str, Any], dataset_ref: str, base_checkpoint: str) -> dict[str, Any]:
        tag = hashlib.sha256(json.dumps([config, dataset_ref, base_checkpoint],
                                        sort_keys=True).encode()).hexdigest()[:10]
        return {"evidence": "EXPLANATION", "status": "COMPLETED",
                "provider_job_id": None, "checkpoint": f"stub-ckpt-{tag}"}

    def evaluate(self, checkpoint: str, suite: str, split: str, seed: int) -> dict[str, Any]:
        base = 0.30 + 0.30 * _unit(checkpoint, suite, split, seed)
        bonus = 0.08 if checkpoint.startswith("stub-ckpt-") else 0.0
        return {"evidence": "EXPLANATION", "checkpoint": checkpoint, "suite": suite,
                "split": split, "seed": seed, "score": round(base + bonus, 4)}


def validate_qlora(config: dict[str, Any]) -> list[str]:
    """Return a list of violations (empty means the config is inside the search space)."""
    errors: list[str] = []
    expected = set(QLORA_SPACE) | {"target_modules"}
    if missing := expected - config.keys():
        errors.append(f"missing keys: {sorted(missing)}")
    if extra := config.keys() - expected:
        errors.append(f"unknown keys: {sorted(extra)}")
    for key in ("rank", "alpha", "sequence_length", "batch_size", "gradient_accumulation_steps"):
        if key in config and config[key] not in QLORA_SPACE[key]:
            errors.append(f"{key}={config[key]!r} not in {QLORA_SPACE[key]}")
    for key in ("dropout", "learning_rate", "epochs"):
        if key in config:
            lo, hi = QLORA_SPACE[key]
            v = config[key]
            if isinstance(v, bool) or not isinstance(v, int | float) or not isfinite(v) \
                    or not lo <= v <= hi:
                errors.append(f"{key}={v!r} outside [{lo}, {hi}]")
    mods = config.get("target_modules")
    if "target_modules" in config and (
        not isinstance(mods, list) or not mods or not set(mods) <= set(TARGET_MODULES)
    ):
        errors.append(f"target_modules must be a non-empty subset of {TARGET_MODULES}")
    return errors


def promotion_gate(champion: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed promotion decision over two evaluation results.

    Requires verified evidence on both sides, identical suite/split/seed, a
    relative improvement of at least 10%, and regression under 0.05 absolute.
    """
    reasons: list[str] = []
    try:
        c, n = float(champion["score"]), float(candidate["score"])
    except (KeyError, TypeError, ValueError):
        return {"decision": "REJECT", "reasons": ["scores missing or non-numeric"]}
    if not (isfinite(c) and isfinite(n)):
        return {"decision": "REJECT", "reasons": ["scores must be finite"]}
    for side, res in (("champion", champion), ("candidate", candidate)):
        if res.get("evidence") not in VERIFIED:
            reasons.append(f"{side} evidence {res.get('evidence')!r} is not verified")
    for key in ("suite", "split", "seed"):
        if champion.get(key) != candidate.get(key):
            reasons.append(f"{key} differs between champion and candidate")
    delta = n - c
    relative = delta / abs(c) if c else None
    if (relative is None and n <= 0) or (relative is not None and relative < MIN_RELATIVE_IMPROVEMENT):
        reasons.append("minimum relative improvement not met")
    if max(0.0, -delta) > MAX_ABSOLUTE_REGRESSION:
        reasons.append("maximum regression exceeded")
    return {"decision": "REJECT" if reasons else "PROMOTE", "reasons": reasons,
            "champion_score": c, "candidate_score": n, "relative_improvement": relative}


def make_tools(backend: ExperimentBackend, seed: int = 7,
               suite: str = "AgentGym/AgentEval") -> dict[str, Callable[..., Any]]:
    """Bind deterministic tools to an experiment backend. Keyed by role."""

    def run_benchmark(checkpoint: str, split: str = "train") -> dict[str, Any]:
        """Benchmark a checkpoint on the TRAIN split and return score + trajectory refs."""
        if split != "train":
            return {"error": "benchmark may only use the train split; held-out data is sealed"}
        return backend.benchmark(checkpoint, suite, seed)

    def validate_qlora_config(config: dict[str, Any]) -> dict[str, Any]:
        """Check a QLoRA config against the fixed search space. Returns {valid, errors}."""
        errors = validate_qlora(config)
        return {"valid": not errors, "errors": errors}

    def launch_training(config: dict[str, Any], dataset_ref: str, base_checkpoint: str) -> dict[str, Any]:
        """Launch exactly one training job with a validated config; returns provider result."""
        if errors := validate_qlora(config):
            return {"status": "BLOCKED", "errors": errors}
        return backend.train(config, dataset_ref, base_checkpoint)

    def evaluate_checkpoint(checkpoint: str, split: str = "heldout") -> dict[str, Any]:
        """Evaluate a checkpoint on a sealed split ('heldout' or 'regression'); scores only."""
        if split not in {"heldout", "regression"}:
            return {"error": "split must be 'heldout' or 'regression'"}
        return backend.evaluate(checkpoint, suite, split, seed)

    def decide_promotion(champion: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        """Run the deterministic promotion gate over two evaluate_checkpoint results."""
        return promotion_gate(champion, candidate)

    return {"run_benchmark": run_benchmark, "validate_qlora_config": validate_qlora_config,
            "launch_training": launch_training, "evaluate_checkpoint": evaluate_checkpoint,
            "decide_promotion": decide_promotion}
