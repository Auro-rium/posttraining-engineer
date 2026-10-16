"""Deterministic post-training rules. Nothing here calls an LLM.

Agents may propose; only this code validates hyperparameters and datasets, enforces the
budget and the sealed held-out boundary, and decides promotion. Every measurement
carries an evidence label so simulated numbers can never be mistaken for measured ones.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
import math
from math import isfinite
from pathlib import Path
from typing import Any, Literal, Protocol

from ..config import Settings
from ..observability.events import Journal, emit
from . import games as games_mod
from ..tools import budget as budget_mod
from ..tools import dataset as dataset_mod

Evidence = Literal["LIVE", "PRIOR_VERIFIED_RUN", "EXPLANATION"]
VERIFIED: frozenset[str] = frozenset({"LIVE", "PRIOR_VERIFIED_RUN"})

# Bounds follow Nebius' SupervisedHParameters schema, tightened for cost control.
# name: (low, high, integer?) inclusive. LoRA is always on.
HPARAM_SPACE: dict[str, tuple[float, float, bool]] = {
    "lora_r": (8, 128, True),
    "lora_alpha": (8, 256, True),
    "lora_dropout": (0.0, 0.1, False),
    "learning_rate": (1e-6, 5e-4, False),
    "n_epochs": (1, 5, True),
    "batch_size": (1, 64, True),
    "context_length": (8192, 32768, True),
}
OPTIONAL_HPARAMS: dict[str, tuple[float, float, bool]] = {
    "warmup_ratio": (0.0, 0.2, False),
    "weight_decay": (0.0, 0.1, False),
}
MIN_RELATIVE_IMPROVEMENT = 0.10
MIN_Z = 1.645  # one-sided 5% two-proportion test: the gain must not be sampling noise
MAX_ABSOLUTE_REGRESSION = 0.05


class ExperimentBackend(Protocol):
    """Where benchmarks, training and evaluation actually run (see ``nebius.NebiusBackend``).

    Real implementations return provider-measured numbers labelled ``LIVE``.
    """

    def benchmark(self, checkpoint: str, suite: str, seed: int) -> dict[str, Any]: ...
    def train(self, config: dict[str, Any], dataset_path: Path, base_model: str) -> dict[str, Any]: ...
    def evaluate(self, checkpoint: str, suite: str, split: str, seed: int) -> dict[str, Any]: ...
    def job_status(self, job_id: str) -> dict[str, Any]: ...
    def job_events(self, job_id: str, limit: int = 50) -> dict[str, Any]: ...


def _unit(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class LocalStubBackend:
    """Deterministic simulation for offline runs. Every result is labelled ``EXPLANATION``."""

    def benchmark(self, checkpoint: str, suite: str, seed: int) -> dict[str, Any]:
        score = round(0.30 + 0.30 * _unit(checkpoint, suite, "train", seed), 4)
        return {"evidence": "EXPLANATION", "checkpoint": checkpoint, "suite": suite,
                "split": "train", "seed": seed, "score": score,
                "failures": [{"task_id": f"stub-{i}", "expected": None, "got": None}
                             for i in range(3)]}

    def train(self, config: dict[str, Any], dataset_path: Path, base_model: str) -> dict[str, Any]:
        tag = hashlib.sha256(json.dumps([config, dataset_path.name, base_model],
                                        sort_keys=True).encode()).hexdigest()[:10]
        return {"evidence": "EXPLANATION", "status": "succeeded",
                "provider_job_id": f"stub-job-{tag}", "checkpoint": f"stub-ckpt-{tag}"}

    def evaluate(self, checkpoint: str, suite: str, split: str, seed: int) -> dict[str, Any]:
        base = 0.30 + 0.30 * _unit(checkpoint, suite, split, seed)
        bonus = 0.08 if checkpoint.startswith("stub-ckpt-") else 0.0
        return {"evidence": "EXPLANATION", "checkpoint": checkpoint, "suite": suite,
                "split": split, "seed": seed, "score": round(base + bonus, 4)}

    def job_status(self, job_id: str) -> dict[str, Any]:
        return {"evidence": "EXPLANATION", "provider_job_id": job_id, "status": "succeeded",
                "checkpoint": job_id.replace("job", "ckpt")}

    def job_events(self, job_id: str, limit: int = 50) -> dict[str, Any]:
        return {"evidence": "EXPLANATION", "provider_job_id": job_id,
                "events": [{"level": "info", "message": "simulated job; no real logs exist"}]}


def validate_hparams(config: dict[str, Any]) -> list[str]:
    """Violations of the fixed search space (empty list means valid)."""
    errors: list[str] = []
    allowed = {**HPARAM_SPACE, **OPTIONAL_HPARAMS}
    if missing := set(HPARAM_SPACE) - config.keys():
        errors.append(f"missing keys: {sorted(missing)}")
    if extra := config.keys() - (set(allowed) | {"packing"}):
        errors.append(f"unknown keys (lora is always on): {sorted(extra)}")
    for key, (lo, hi, is_int) in allowed.items():
        if key not in config:
            continue
        v = config[key]
        ok = isinstance(v, int | float) and not isinstance(v, bool) and isfinite(v) and lo <= v <= hi
        if ok and is_int and int(v) != v:
            ok = False
        if not ok:
            errors.append(f"{key}={v!r} must be {'an integer ' if is_int else ''}in [{lo}, {hi}]")
    if "packing" in config and not isinstance(config["packing"], bool):
        errors.append("packing must be a boolean")
    r, a = config.get("lora_r"), config.get("lora_alpha")
    if isinstance(r, int | float) and isinstance(a, int | float) and a > 4 * r:
        errors.append("lora_alpha must be <= 4 * lora_r")
    return errors


def promotion_gate(champion: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed promotion decision over two evaluation results.

    Requires verified evidence on both sides, identical suite/split/seed, a relative
    improvement of at least 10%, and regression under 0.05 absolute.
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
    for key in ("suite", "split", "seed", "judged_by"):  # judged_by: never compare two judges
        if champion.get(key) != candidate.get(key):
            reasons.append(f"{key} differs between champion and candidate")
    delta = n - c
    relative = delta / abs(c) if c else None
    if (relative is None and n <= 0) or (relative is not None and relative < MIN_RELATIVE_IMPROVEMENT):
        reasons.append("minimum relative improvement not met")
    if max(0.0, -delta) > MAX_ABSOLUTE_REGRESSION:
        reasons.append("maximum regression exceeded")
    z = _improvement_z(champion, candidate, c, n)
    if z is not None and z < MIN_Z:
        reasons.append(f"improvement not statistically significant (z={z:.2f} < {MIN_Z}, "
                       f"n={champion['n']}/{candidate['n']})")
    return {"decision": "REJECT" if reasons else "PROMOTE", "reasons": reasons,
            "champion_score": c, "candidate_score": n, "relative_improvement": relative,
            "z": None if z is None else round(z, 3)}


def _improvement_z(champion: dict[str, Any], candidate: dict[str, Any],
                   c: float, n: float) -> float | None:
    """One-sided two-proportion z for candidate > champion; None if sizes are not usable.

    Scores are accuracies (fractions of ``n`` tasks). Without integer ``n`` on both sides, or with
    scores outside [0, 1], no significance test applies and the other rules decide alone.
    """
    n1, n2 = champion.get("n"), candidate.get("n")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (n1, n2)):
        return None
    if not (0.0 <= c <= 1.0 and 0.0 <= n <= 1.0):
        return None
    pooled = (c * n1 + n * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return math.inf if n > c else 0.0
    return (n - c) / se


def make_tools(backend: ExperimentBackend, settings: Settings,
               suite: str = "AgentGym/AgentEval",
               journal: Journal | None = None) -> dict[str, Callable[..., Any]]:
    """Bind the deterministic tools to a backend and settings.

    With a ``journal`` the tools emit score/gate/budget/job events under ``CURRENT_RUN``;
    emission is best-effort and never changes a tool's result.
    """
    seed = settings.seed

    def _score(res: dict[str, Any], split: str, checkpoint: str) -> None:
        if isinstance(res, dict) and "score" in res:
            emit(journal, "score", {
                "split": res.get("split", split), "checkpoint": res.get("checkpoint", checkpoint),
                "value": res.get("score"), "n": res.get("n"), "evidence": res.get("evidence")})

    def _budget(est: Any) -> None:
        if isinstance(est, dict) and "error" not in est:
            emit(journal, "budget.update", {
                "spent_usd": est.get("spent_usd"), "cap_usd": est.get("cap_usd"),
                "estimate_usd": est.get("estimated_cost_usd"),
                "price_known": est.get("price_per_mtok") is not None})

    def _job(res: Any) -> None:
        if isinstance(res, dict) and res.get("provider_job_id"):
            emit(journal, "job.update", {
                "job_id": res.get("provider_job_id"), "status": res.get("status"),
                "trained_steps": res.get("trained_steps"), "total_steps": res.get("total_steps"),
                "trained_tokens": res.get("trained_tokens"), "evidence": res.get("evidence")})

    def run_benchmark(checkpoint: str, split: str = "train") -> dict[str, Any]:
        """Benchmark a checkpoint ('base' = the base model) on the TRAIN split; returns score + failures."""
        if split != "train":
            return {"error": "benchmark may only use the train split; held-out data is sealed"}
        res = backend.benchmark(checkpoint, suite, seed)
        _score(res, "train", checkpoint)
        return res

    def validate_hparams_tool(config: dict[str, Any]) -> dict[str, Any]:
        """Check LoRA hyperparameters against the fixed search space. Returns {valid, errors}."""
        errors = validate_hparams(config)
        return {"valid": not errors, "errors": errors}

    def verify_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Verify chat-format SFT rows (schema, allow-listed tools, no sealed refs, no duplicates)."""
        return dataset_mod.verify_rows(rows, max_rows=settings.max_dataset_rows)

    def save_dataset(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Verify rows and, only if they pass, save them as '<name>.jsonl'. Returns the dataset ref."""
        ref = name if name.endswith(".jsonl") else f"{name}.jsonl"
        return dataset_mod.save_rows(settings.data_dir, ref, rows, settings.max_dataset_rows)

    def generate_game_sft(name: str, n_rows: int = 600, seed: int = 100) -> dict[str, Any]:
        """Generate rule-correct SFT rows (max 1200) from the four games, verify and save them.

        Rows are computed by code (never written by a model), balanced over the games, use
        paraphrased prompts, and never contain a state from any evaluation split.
        """
        states = games_mod.suite_states(settings.suite_dir)
        if not states:
            return {"ok": False, "errors": [
                "evaluation suite files carry no 'state' keys; regenerate them with "
                "scripts/make_game_suites.py so training data can exclude eval states"]}
        rows, counts = games_mod.generate_sft(n_rows, seed, states, settings.max_dataset_rows)
        ref = name if name.endswith(".jsonl") else f"{name}.jsonl"
        report = dataset_mod.save_rows(settings.data_dir, ref, rows, settings.max_dataset_rows)
        return {**report, "by_game": counts, "excluded_eval_states": len(states),
                "requested": n_rows}

    def estimate_budget(dataset_ref: str, hyperparameters: dict[str, Any]) -> dict[str, Any]:
        """Estimate trained tokens and cost of a job against the remaining budget."""
        est = budget_mod.estimate(settings, dataset_ref, hyperparameters)
        _budget(est)
        return est

    def launch_training(hyperparameters: dict[str, Any], dataset_ref: str,
                        base_model: str | None = None) -> dict[str, Any]:
        """Launch ONE LoRA fine-tuning job (needs human approval). Re-checks hyperparameters, dataset and budget."""
        if errors := validate_hparams(hyperparameters):
            return {"status": "BLOCKED", "stage": "hyperparameters", "errors": errors}
        report = dataset_mod.verify_file(settings.data_dir, dataset_ref, settings.max_dataset_rows)
        if not report["ok"]:
            return {"status": "BLOCKED", "stage": "dataset", "errors": report["errors"]}
        est = budget_mod.estimate(settings, dataset_ref, hyperparameters)
        if est.get("within_budget") is not True:
            return {"status": "BLOCKED", "stage": "budget", "estimate": est}
        result = backend.train(hyperparameters, dataset_mod.resolve(settings.data_dir, dataset_ref),
                               base_model or settings.base_model)
        if result.get("provider_job_id"):
            cost = est["estimated_cost_usd"] or 0.0  # None when running unpriced
            budget_mod.record_spend(settings, result["provider_job_id"], cost, dataset_ref)
            # no re-read: nothing may raise after a paid launch
            _budget({**est, "spent_usd": round(est["spent_usd"] + cost, 4)})
        _job(result)
        return {**result, "dataset_sha256": report["sha256"], "estimate": est}

    def get_training_status(job_id: str) -> dict[str, Any]:
        """Poll a training job: status, steps, trained tokens and (when done) the checkpoint name."""
        res = backend.job_status(job_id)
        _job(res)
        return res

    def read_training_logs(job_id: str, limit: int = 50) -> dict[str, Any]:
        """Read the provider's event log for a training job (most useful after failures)."""
        res = backend.job_events(job_id, max(1, min(limit, 200)))
        if isinstance(res, dict):
            for ev in res.get("events") or []:
                if isinstance(ev, dict):
                    emit(journal, "job.log", {"level": ev.get("level"),
                                              "message": ev.get("message")})
        return res

    def evaluate_checkpoint(checkpoint: str, split: str = "heldout") -> dict[str, Any]:
        """Evaluate a checkpoint on a sealed split ('heldout' or 'regression'); scores only."""
        if split not in {"heldout", "regression"}:
            return {"error": "split must be 'heldout' or 'regression'"}
        res = backend.evaluate(checkpoint, suite, split, seed)
        _score(res, split, checkpoint)
        return res

    def decide_promotion(champion: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        """Run the deterministic promotion gate over two evaluate_checkpoint results."""
        verdict = promotion_gate(champion, candidate)
        emit(journal, "gate.verdict", verdict)
        return verdict

    validate_hparams_tool.__name__ = "validate_hparams"
    return {"run_benchmark": run_benchmark, "validate_hparams": validate_hparams_tool,
            "verify_dataset": verify_dataset, "save_dataset": save_dataset,
            "generate_game_sft": generate_game_sft,
            "estimate_budget": estimate_budget, "launch_training": launch_training,
            "get_training_status": get_training_status, "read_training_logs": read_training_logs,
            "evaluate_checkpoint": evaluate_checkpoint, "decide_promotion": decide_promotion}
