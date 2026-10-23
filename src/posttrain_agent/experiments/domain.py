"""Deterministic post-training rules. Nothing here calls an LLM.

Agents may propose; only this code validates hyperparameters and datasets, enforces the
budget and the sealed held-out boundary, and decides promotion. Every measurement
carries an evidence label so simulated numbers can never be mistaken for measured ones.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
from collections.abc import Callable
from math import isfinite
from pathlib import Path
from typing import Any, Literal, Protocol

from ..config import Settings
from ..observability.events import CURRENT_RUN, Journal, emit
from ..tools import budget as budget_mod
from ..tools import dataset as dataset_mod
from . import games as games_mod
from . import ledger as ledger_mod
from . import pretokenize as pretokenize_mod
from . import runstate as rs
from .gate import (  # noqa: F401
    MAX_ABSOLUTE_REGRESSION,
    MIN_RELATIVE_IMPROVEMENT,
    MIN_Z,
    promotion_gate,
)

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
# the gate itself lives in gate.py; re-exported here for existing imports
MIN_STEPS_BLOCK = 50  # fewer optimizer steps than this cannot teach a format: refuse
MIN_STEPS_WARN = 100


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


def estimate_steps(config: dict[str, Any], total_tokens: float, avg_row_tokens: float) -> int | None:
    """Approximate optimizer steps: ceil(tokens * epochs / (batch_size * tokens_per_sequence)).

    With packing a sequence is a full ``context_length`` window; without it, one row. ``total_tokens``
    is per epoch. None when the inputs are unusable."""
    try:
        per_seq = config["context_length"] if config.get("packing") else avg_row_tokens
        denom = float(config["batch_size"]) * float(per_seq)
        if denom <= 0 or total_tokens <= 0:
            return None
        return max(1, math.ceil(total_tokens * float(config["n_epochs"]) / denom))
    except (KeyError, TypeError, ValueError):
        return None


def step_findings(config: dict[str, Any], total_tokens: float,
                  avg_row_tokens: float) -> tuple[int | None, list[str], list[str]]:
    """(steps, blocking errors, warnings) for the optimizer-step guard."""
    steps = estimate_steps(config, total_tokens, avg_row_tokens)
    if steps is None:
        return None, [], []
    hint = ("raise n_epochs, lower batch_size (1-4 suits a few hundred short rows), or set "
            "packing=false")
    if steps < MIN_STEPS_BLOCK:
        return steps, [f"only ~{steps} optimizer steps (< {MIN_STEPS_BLOCK}): too few to learn; "
                       + hint], []
    if steps < MIN_STEPS_WARN:
        return steps, [], [f"only ~{steps} optimizer steps (< {MIN_STEPS_WARN}); {hint}"]
    return steps, [], []


def cost_summary(settings: Settings, dataset_ref: str, hyperparameters: dict[str, Any],
                 est: dict[str, Any] | None = None) -> dict[str, Any]:
    """Human-facing cost facts for one job: estimated trained tokens vs the cap, optimizer steps and
    an explicit UNPRICED / known-USD line. Display only; it changes no gate or approval logic."""
    est = est if est is not None else budget_mod.estimate(settings, dataset_ref, hyperparameters)
    if "error" in est:
        return {"error": est["error"], "cost_line": f"cost UNKNOWN: {est['error']}"}
    info = budget_mod.training_token_info(settings, dataset_ref)
    steps = None
    if "error" not in info:
        steps = estimate_steps(hyperparameters, info["total_tokens"], info["avg_row_tokens"])
    tokens, spent_t = est.get("estimated_trained_tokens"), est.get("spent_trained_tokens", 0)
    cap = est.get("max_trained_tokens")
    if est.get("price_per_mtok") is None:
        usd = "UNPRICED: dollar cost UNKNOWN (no price configured)"
        cap_txt = (f"{tokens} of cap {cap} (cumulative {spent_t} + {tokens} = {spent_t + tokens})"
                   if cap is not None else f"{tokens} (no token cap: launch needs a price or opt-in)")
    else:
        usd = (f"known USD: ~${est.get('estimated_cost_usd')} for this job; spent "
               f"${est.get('spent_usd')} of ${est.get('cap_usd')} cap")
        cap_txt = f"{tokens}"
    return {"estimated_trained_tokens": tokens, "max_trained_tokens": cap,
            "spent_trained_tokens": spent_t, "estimated_steps": steps,
            "epochs": est.get("epochs"), "usd": usd,
            "cost_line": f"trained tokens: {cap_txt}; optimizer steps: ~{steps}; {usd}"}


def validate_hparams(config: dict[str, Any], *, total_tokens: float | None = None,
                     avg_row_tokens: float | None = None) -> list[str]:
    """Violations of the fixed search space (empty list means valid).

    With the dataset's per-epoch ``total_tokens`` and ``avg_row_tokens`` it also enforces the
    optimizer-step floor (see ``step_findings``)."""
    errors = _validate_space(config)
    if not errors and total_tokens is not None and avg_row_tokens is not None:
        errors += step_findings(config, total_tokens, avg_row_tokens)[1]
    return errors


def _validate_space(config: dict[str, Any]) -> list[str]:
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


def make_tools(backend: ExperimentBackend, settings: Settings,
               suite: str = "AgentGym/AgentEval",
               journal: Journal | None = None,
               sleep: Callable[[float], None] = time.sleep) -> dict[str, Callable[..., Any]]:
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

    def _ledger(res: dict[str, Any], split: str, checkpoint: str, job: str | None) -> None:
        """Append every real evaluation attempt (ok or structured error) to the eval ledger."""
        if not isinstance(res, dict) or res.get("evidence") == "EXPLANATION":
            return  # simulated numbers never enter the ledger
        try:
            ledger_mod.append(settings.state_dir, run=CURRENT_RUN.get() or None,
                              checkpoint=checkpoint, split=split, result=res, job=job,
                              suite=suite, seed=seed)
        except OSError:
            pass  # best-effort: the ledger must never change a tool's result

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

    def _hp(value: Any) -> Any:
        """Accept hyperparameters as an object or as a JSON string of one (models send both)."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                return value
        return value

    def validate_hparams_tool(
        hyperparameters: dict[str, Any] | str | None = None, dataset_ref: str | None = None,
        lora_r: Any = None, lora_alpha: Any = None, lora_dropout: Any = None,
        learning_rate: Any = None, n_epochs: Any = None, batch_size: Any = None,
        context_length: Any = None, warmup_ratio: Any = None, weight_decay: Any = None,
        packing: Any = None,
    ) -> dict[str, Any]:
        """Check LoRA hyperparameters against the fixed search space. Returns {valid, errors}.

        Pass them nested as `hyperparameters={...}` (same shape as estimate_budget and
        launch_training) or as flat arguments (`lora_r=16, ...`); flat values override nested.
        `hyperparameters` may also be a JSON string. With `dataset_ref` it also estimates the
        optimizer steps and BLOCKS under ~50 steps (warns under ~100): `steps`, `warnings`.
        """
        # Not named `config`: LangChain reserves that argument name for its RunnableConfig and
        # silently drops it from the model's call, which crashed this tool before.
        flat = {"lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
                "learning_rate": learning_rate, "n_epochs": n_epochs, "batch_size": batch_size,
                "context_length": context_length, "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay, "packing": packing}
        hyperparameters = _hp(hyperparameters)
        if hyperparameters is not None and not isinstance(hyperparameters, dict):
            return {"valid": False, "errors": ["hyperparameters must be an object"]}
        merged = {**(hyperparameters or {}), **{k: v for k, v in flat.items() if v is not None}}
        if not merged:
            return {"valid": False, "errors": [
                "no hyperparameters received: pass them as hyperparameters={...} or as flat "
                "arguments (lora_r=..., ...); an argument named `config` is dropped by the tool "
                "runtime"]}
        errors = validate_hparams(merged)
        out: dict[str, Any] = {"valid": not errors, "errors": errors, "checked": merged}
        if dataset_ref and not errors:
            info = budget_mod.training_token_info(settings, dataset_ref)
            if "error" in info:
                out["warnings"] = [f"step guard not applied: {info['error']}"]
            else:
                steps, step_errors, warns = step_findings(
                    merged, info["total_tokens"], info["avg_row_tokens"])
                out.update(valid=not step_errors, errors=step_errors, warnings=warns, steps=steps,
                           step_basis=info["source"])
        return out

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

    def estimate_budget(dataset_ref: str, hyperparameters: dict[str, Any] | str) -> dict[str, Any]:
        """Estimate trained tokens and cost of a job against the remaining budget."""
        hyperparameters = _hp(hyperparameters)
        if not isinstance(hyperparameters, dict):
            return {"error": "hyperparameters must be an object (or a JSON string of one)"}
        est = budget_mod.estimate(settings, dataset_ref, hyperparameters)
        _budget(est)
        return est

    def launch_training(hyperparameters: dict[str, Any] | str, dataset_ref: str,
                        base_model: str | None = None) -> dict[str, Any]:
        """Launch ONE LoRA fine-tuning job (needs human approval). Re-checks hyperparameters, dataset,
        the chat-template check (check_chat_template must have passed for this exact dataset), the
        optimizer-step floor and the budget. Trains on the pretokenized file."""
        hyperparameters = _hp(hyperparameters)
        if not isinstance(hyperparameters, dict):
            return {"status": "BLOCKED", "stage": "hyperparameters",
                    "errors": ["hyperparameters must be an object (or a JSON string of one)"]}
        if errors := validate_hparams(hyperparameters):
            return {"status": "BLOCKED", "stage": "hyperparameters", "errors": errors}
        report = dataset_mod.verify_file(settings.data_dir, dataset_ref, settings.max_dataset_rows)
        if not report["ok"]:
            return {"status": "BLOCKED", "stage": "dataset", "errors": report["errors"]}
        if not dataset_mod.check_marker_ok(settings.data_dir, dataset_ref, report["sha256"]):
            return {"status": "BLOCKED", "stage": "template_check", "errors": [
                f"no passing check_chat_template for {dataset_ref} with sha256 {report['sha256']}; "
                "run check_chat_template(dataset_ref) on the saved dataset first (any change to the "
                "file invalidates an earlier check)"]}
        info = budget_mod.training_token_info(settings, dataset_ref)
        if "error" in info:
            return {"status": "BLOCKED", "stage": "dataset", "errors": [info["error"]]}
        steps, step_errors, _ = step_findings(hyperparameters, info["total_tokens"],
                                              info["avg_row_tokens"])
        if step_errors:
            return {"status": "BLOCKED", "stage": "steps", "steps": steps, "errors": step_errors}
        est = budget_mod.estimate(settings, dataset_ref, hyperparameters)
        if est.get("within_budget") is not True:
            return {"status": "BLOCKED", "stage": "budget", "estimate": est}
        source = dataset_mod.resolve(settings.data_dir, dataset_ref)
        pre = pretokenize_mod.pretok_path(source)
        train_path = pre if pre.exists() else source
        key = rs.intent_key(report["sha256"], hyperparameters)
        prior = rs.find_by_key(settings.state_dir, key)
        if prior and str(prior.get("status", "")).lower() not in rs.TERMINAL_FAILED:
            return {"status": "BLOCKED", "stage": "duplicate", "errors": [
                "a job with this exact dataset sha256 and hyperparameters was already launched "
                f"(job {prior.get('job_id') or 'unknown: launch intent without a job id'}, status "
                f"{prior.get('status')}); wait_for_training on it instead of launching again"],
                "provider_job_id": prior.get("job_id")}
        intent = {"run": CURRENT_RUN.get() or None, "key": key, "dataset_ref": dataset_ref,
                  "dataset_sha256": report["sha256"], "hparams": hyperparameters,
                  "hparams_hash": rs.hparams_hash(hyperparameters)}
        rs.write_run_state(settings.state_dir, **intent, job_id=None, status="LAUNCHING",
                           checkpoint=None, trained_tokens=None)
        try:
            result = backend.train(hyperparameters, train_path, base_model or settings.base_model)
        except Exception:
            rs.write_run_state(settings.state_dir, **intent, job_id=None, status="LAUNCH_FAILED",
                               checkpoint=None, trained_tokens=None)
            raise
        if result.get("provider_job_id"):
            rs.write_run_state(settings.state_dir, **intent, job_id=result["provider_job_id"],
                               status=str(result.get("status") or "queued"), checkpoint=None,
                               trained_tokens=None)
        else:
            rs.write_run_state(settings.state_dir, **intent, job_id=None, status="LAUNCH_FAILED",
                               checkpoint=None, trained_tokens=None)
        if result.get("provider_job_id"):
            cost = est["estimated_cost_usd"] or 0.0  # None when running unpriced
            budget_mod.record_spend(settings, result["provider_job_id"], cost, dataset_ref,
                                    est.get("estimated_trained_tokens"))
            # no re-read: nothing may raise after a paid launch
            _budget({**est, "spent_usd": round(est["spent_usd"] + cost, 4)})
        _job(result)
        return {**result, "dataset_sha256": report["sha256"], "estimate": est,
                "trained_file": train_path.name, "estimated_steps": steps,
                "cost": cost_summary(settings, dataset_ref, hyperparameters, est)}

    def get_training_status(job_id: str) -> dict[str, Any]:
        """Poll a training job: status, steps, trained tokens and (when done) the checkpoint name."""
        res = backend.job_status(job_id)
        _job(res)
        with contextlib.suppress(OSError):
            if isinstance(res, dict) and res.get("status"):
                _record_status(job_id, res)
        # replace the launch estimate with the provider's actual trained tokens; bookkeeping
        # must never break status polling
        with contextlib.suppress(Exception):
            if isinstance(res, dict):
                budget_mod.reconcile(settings, job_id, res.get("trained_tokens"))
        return res

    def _record_status(job_id: str, res: dict[str, Any]) -> None:
        prev = rs.find_job(settings.state_dir, job_id) or {}
        rs.write_run_state(
            settings.state_dir, run=prev.get("run") or CURRENT_RUN.get() or None,
            key=prev.get("key"), job_id=job_id, dataset_ref=prev.get("dataset_ref"),
            dataset_sha256=prev.get("dataset_sha256"), hparams=prev.get("hparams"),
            hparams_hash=prev.get("hparams_hash"), status=str(res.get("status") or "unknown"),
            checkpoint=res.get("checkpoint"), trained_tokens=res.get("trained_tokens"))

    def wait_for_training(job_id: str, timeout_s: int = 600, poll_s: int = 20) -> dict[str, Any]:
        """BLOCK until the training job is terminal (succeeded/failed/cancelled) or timeout_s passes,
        polling every poll_s seconds inside this tool (no model calls). Records the run state
        (job, dataset sha, hyperparameters, status, checkpoint, trained tokens) and reconciles the
        actual trained tokens into the spend log. On timeout returns timed_out=true with the last
        status: call it again with the same job id. SUCCEEDED means a checkpoint is present."""
        timeout_s = max(1, min(int(timeout_s), 3600))
        poll_s = max(1, min(int(poll_s), 120))
        waited, res = 0, {}
        while True:
            res = backend.job_status(job_id)
            _job(res)
            if not isinstance(res, dict) or res.get("error") and "status" not in res:
                return {"job_id": job_id, "terminal": False, "timed_out": False, "status": "ERROR",
                        "error": (res or {}).get("error") if isinstance(res, dict) else str(res)}
            status = str(res.get("status") or "").lower()
            if status in rs.TERMINAL or waited >= timeout_s:
                break
            sleep(poll_s)
            waited += poll_s
        with contextlib.suppress(Exception):
            budget_mod.reconcile(settings, job_id, res.get("trained_tokens"))
        with contextlib.suppress(OSError):
            _record_status(job_id, res)
        terminal = status in rs.TERMINAL
        out = {**res, "job_id": job_id, "terminal": terminal, "timed_out": not terminal,
               "waited_s": waited}
        if status == rs.SUCCEEDED and not res.get("checkpoint"):
            out.update(status="FAILED_NO_CHECKPOINT", terminal=True,
                       error="provider reports succeeded but returned no checkpoint: not usable")
        return out

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
        """Evaluate a checkpoint on a sealed split ('heldout' or 'regression'; 'heldout_phrasing' is
        informational only and never gates); scores only."""
        if split not in {"heldout", "regression", "heldout_phrasing"}:
            return {"error": "split must be 'heldout', 'regression' or 'heldout_phrasing'"}
        job = None
        if checkpoint in {"candidate", "latest"}:
            rec = rs.latest_succeeded(settings.state_dir, CURRENT_RUN.get())
            if rec is None:
                return {"error": "no succeeded training job with a checkpoint is recorded for "
                                 "this run; wait_for_training must finish first",
                        "stage": "candidate"}
            checkpoint, job = rec["checkpoint"], rec.get("job_id")
        if checkpoint.startswith("ftjob-"):  # a job id means that job's final checkpoint
            job = checkpoint
            st = backend.job_status(checkpoint)
            if not st.get("checkpoint"):  # an "error": null key is normal on healthy jobs
                res = {"error": f"job {checkpoint} has no finished checkpoint",
                       "stage": "checkpoint", "detail": str(st.get("error") or st.get("status"))}
                _ledger(res, split, checkpoint, job)
                return res
            checkpoint = st["checkpoint"]
        res = backend.evaluate(checkpoint, suite, split, seed)
        _score(res, split, checkpoint)
        _ledger(res, split, checkpoint, job)
        return res

    def decide_promotion(champion: dict[str, Any], candidate: dict[str, Any],
                         champion_regression: dict[str, Any] | None = None,
                         candidate_regression: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the deterministic promotion gate over two held-out evaluate_checkpoint results;
        pass both regression-split results too so forgetting is checked."""
        verdict = promotion_gate(champion, candidate, champion_regression=champion_regression,
                                 candidate_regression=candidate_regression)
        emit(journal, "gate.verdict", verdict)
        return verdict

    def _resolve_ref(ref: str) -> tuple[str, str | None]:
        """(checkpoint name, job id) for 'base', 'candidate', a job id or a checkpoint name."""
        if ref in {"candidate", "latest"}:
            rec = rs.latest_succeeded(settings.state_dir, CURRENT_RUN.get())
            return (rec["checkpoint"], rec.get("job_id")) if rec else (ref, None)
        if ref.startswith("ftjob-"):
            rec = rs.find_job(settings.state_dir, ref)
            if rec and rec.get("checkpoint"):
                return rec["checkpoint"], ref
            for r in reversed(ledger_mod.read(settings.state_dir)):
                if r.get("job") == ref and r.get("status") == "ok":
                    return r["checkpoint"], ref
            return ref, ref
        for r in reversed(rs.run_states(settings.state_dir)):
            if r.get("checkpoint") == ref:
                return ref, r.get("job_id")
        return ref, None

    def run_gate(champion_ref: str, candidate_ref: str = "candidate") -> dict[str, Any]:
        """Deterministic promotion gate over the evaluation LEDGER: reads the held-out and regression
        results of both checkpoints that evaluate_checkpoint stored (no hand-copied scores) and
        calls the gate with require_regression=True. Refs: 'base', a checkpoint name, an ftjob id
        or 'candidate'. If a split is missing, returns an error naming it: evaluate it first."""
        champ, _cj = _resolve_ref(champion_ref)
        cand, cand_job = _resolve_ref(candidate_ref)
        sd = settings.state_dir
        got: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for split in ("heldout", "regression"):
            c = ledger_mod.latest_ok(sd, checkpoint=cand, split=split)
            if c is None and cand_job:
                c = ledger_mod.latest_ok(sd, job=cand_job, split=split)
            ch = (ledger_mod.latest_matching(sd, checkpoint=champ, split=split, like=c["result"])
                  if c else None) or ledger_mod.latest_ok(sd, checkpoint=champ, split=split)
            for who, rec in (("candidate", c), ("champion", ch)):
                if rec is None:
                    missing.append(f"{who} {split}")
            if c and ch:
                got[f"c_{split}"], got[f"h_{split}"] = c["result"], ch["result"]
        if missing:
            return {"error": "evaluation ledger lacks ok results for: " + ", ".join(missing)
                             + "; run evaluate_checkpoint for them (re-dispatch the evaluator with "
                               "the job id) before gating", "stage": "ledger", "missing": missing,
                    "champion": champ, "candidate": cand}
        verdict = promotion_gate(
            got["h_heldout"], got["c_heldout"], champion_regression=got["h_regression"],
            candidate_regression=got["c_regression"], require_regression=True)
        verdict = {**verdict, "champion": champ, "candidate": cand}
        with contextlib.suppress(OSError):
            rs.record_gate(sd, run=CURRENT_RUN.get() or None, champion=champ, candidate=cand,
                           candidate_job=cand_job, decision=verdict.get("decision"))
        emit(journal, "gate.verdict", verdict)
        return verdict

    def abort_experiment(reason: str) -> dict[str, Any]:
        """Record that the current experiment is being abandoned without a gate verdict (training
        failed, human rejected, budget gone...). The only legal way to end a run with an unevaluated
        candidate."""
        if not str(reason).strip():
            return {"error": "reason must not be empty"}
        rs.record_abort(settings.state_dir, CURRENT_RUN.get(), str(reason)[:500])
        return {"aborted": True, "reason": str(reason)[:500]}

    def inspect_dataset(dataset_ref: str, n: int = 3, game: str | None = None) -> dict[str, Any]:
        """Look at a saved dataset: row/state/style counts per game and n sample rows (optionally of
        one game) as serving prompt text plus the exact training target. Read it before approving."""
        try:
            return dataset_mod.inspect_dataset(settings.data_dir, dataset_ref, n, game,
                                               settings.max_dataset_rows)
        except ValueError as exc:
            return {"ok": False, "errors": [str(exc)]}

    def check_chat_template(dataset_ref: str) -> dict[str, Any]:
        """Prove a saved dataset trains the right thing: pretokenize it with the base model's chat
        template, assert the tool call is the ONLY unmasked span, the prefix equals the serving
        prompt and the call parses. Writes the marker launch_training requires (bound to the
        dataset's sha256). Returns supervised-token fraction and per-game counts."""
        try:
            return dataset_mod.check_chat_template(settings.data_dir, dataset_ref,
                                                   settings.max_dataset_rows)
        except ValueError as exc:
            return {"ok": False, "errors": [str(exc)]}

    validate_hparams_tool.__name__ = "validate_hparams"
    return {"run_benchmark": run_benchmark, "validate_hparams": validate_hparams_tool,
            "verify_dataset": verify_dataset, "save_dataset": save_dataset,
            "generate_game_sft": generate_game_sft, "inspect_dataset": inspect_dataset,
            "check_chat_template": check_chat_template,
            "estimate_budget": estimate_budget, "launch_training": launch_training,
            "get_training_status": get_training_status,
            "wait_for_training": wait_for_training, "run_gate": run_gate,
            "abort_experiment": abort_experiment, "read_training_logs": read_training_logs,
            "evaluate_checkpoint": evaluate_checkpoint, "decide_promotion": decide_promotion}
