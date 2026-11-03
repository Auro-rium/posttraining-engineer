"""Deterministic training-cost estimate and cumulative spend ledger.

Nebius Token Factory publishes no per-token fine-tuning price, so the price is
operator-supplied (``POSTTRAIN_PRICE_PER_MTOK``). Without it the estimate reports
token counts only and ``within_budget`` is ``None``; launching then fails closed.
Token counts come from the base model's tokenizer and chat template (tools schema included),
with a 4x-scaled chars/4 fallback; actual billing is by compute and may differ. Actual
``trained_tokens`` are reconciled into the ledger, and the unpriced token cap is cumulative.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any

from ..config import Settings
from .dataset import load_rows

FALLBACK_FACTOR = 4  # chars/4 undercounts ~3.7x (tools schema + chat template); scale it up


def _row_chars_tokens(row: dict[str, Any]) -> int:
    """Old chars/4 heuristic (messages only), used when no tokenizer is available."""
    chars = 0
    for m in row.get("messages", []):
        chars += len(m.get("content") or "") + 16  # role/format overhead
        for c in m.get("tool_calls") or []:
            fn = c.get("function", {})
            chars += len(fn.get("name", "")) + len(str(fn.get("arguments", "")))
    return max(1, chars // 4)


def _row_tokens(row: dict[str, Any], tok: Any = None) -> int:
    """Tokens of the whole row as the provider's template renders it, tools schema included."""
    if tok is not None:
        with contextlib.suppress(Exception):  # else fall through to the conservative estimate
            from ..experiments.pretokenize import canonical_tool

            tools = [canonical_tool(t) for t in row.get("tools") or []]
            return max(1, tok.count(tok.render(row.get("messages", []), tools, False)))
    return _row_chars_tokens(row) * FALLBACK_FACTOR


def _tokenizer() -> Any:
    try:
        from ..experiments.pretokenize import TokenizerUnavailable, get_tokenizer

        try:
            return get_tokenizer()
        except TokenizerUnavailable:
            return None
    except ImportError:
        return None


def dataset_token_stats(settings: Settings, dataset_ref: str) -> dict[str, Any]:
    """{rows, per_row (token counts), total_tokens, avg_row_tokens, basis} or {error}."""
    try:
        rows = [r for r in load_rows(settings.data_dir, dataset_ref) if isinstance(r, dict)]
    except (OSError, ValueError) as exc:
        return {"error": f"cannot read dataset: {exc}"}
    tok = _tokenizer()
    per_row = [_row_tokens(r, tok) for r in rows]
    return {"rows": len(per_row), "per_row": per_row, "total_tokens": sum(per_row),
            "avg_row_tokens": (sum(per_row) / len(per_row)) if per_row else 0.0,
            "basis": "tokenizer (chat template + tools schema)" if tok is not None
            else f"chars/4 x {FALLBACK_FACTOR} fallback (no tokenizer available)"}


def spend_path(settings: Settings) -> Path:
    return Path(settings.state_dir) / "spend.jsonl"


def _ledger(settings: Settings) -> dict[str, dict[str, Any]]:
    """job_id -> {cost, tokens}: launch records, overridden by reconciled actuals."""
    path = spend_path(settings)
    jobs: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return jobs
    for n, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        key = rec.get("job_id") or f"_anon{n}"
        job = jobs.setdefault(key, {"cost": 0.0, "tokens": 0})
        if rec.get("kind") == "reconcile":
            job["tokens"] = rec["trained_tokens"]
            if rec.get("actual_cost_usd") is not None:
                job["cost"] = rec["actual_cost_usd"]
        else:
            job["cost"] = rec.get("estimated_cost_usd") or 0.0
            job["tokens"] = rec.get("estimated_trained_tokens") or 0
    return jobs


def spent_usd(settings: Settings) -> float:
    return sum(j["cost"] for j in _ledger(settings).values())


def spent_tokens(settings: Settings) -> int:
    """Trained tokens over all launched jobs (actual once reconciled, else the launch estimate)."""
    return sum(j["tokens"] for j in _ledger(settings).values())


def record_spend(settings: Settings, job_id: str | None, cost: float, dataset_ref: str,
                 estimated_tokens: int | None = None) -> None:
    path = spend_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps({"ts": time.time(), "job_id": job_id, "dataset_ref": dataset_ref,
                             "estimated_cost_usd": cost,
                             "estimated_trained_tokens": estimated_tokens}) + "\n")


def reconcile(settings: Settings, job_id: str, trained_tokens: int | None,
              status: str | None = None) -> bool:
    """Replace a launched job's estimate with the provider's actual ``trained_tokens`` (once per
    value). Returns True if a record was written.

    Zero only counts when ``status`` says the job is over (``failed``/``cancelled``): a job the
    provider rejected trained nothing, while a running job reporting 0 has simply not started."""
    if not job_id or not isinstance(trained_tokens, int) or isinstance(trained_tokens, bool):
        return False
    ended = str(status or "").lower() in {"failed", "cancelled", "canceled"}
    if trained_tokens < 0 or (trained_tokens == 0 and not ended):
        return False
    jobs = _ledger(settings)
    if job_id not in jobs or jobs[job_id]["tokens"] == trained_tokens:
        return False
    actual = (round(trained_tokens / 1e6 * settings.price_per_mtok, 4)
              if settings.price_per_mtok is not None else None)
    with spend_path(settings).open("a") as fh:
        fh.write(json.dumps({"ts": time.time(), "kind": "reconcile", "job_id": job_id,
                             "trained_tokens": trained_tokens, "actual_cost_usd": actual}) + "\n")
    return True


def estimate(settings: Settings, dataset_ref: str, hyperparameters: dict[str, Any]) -> dict[str, Any]:
    epochs = hyperparameters.get("n_epochs", 3)
    ctx = hyperparameters.get("context_length", 8192)
    stats = dataset_token_stats(settings, dataset_ref)
    if "error" in stats:
        return stats
    per_row = stats["per_row"]
    trained = sum(min(t, ctx) for t in per_row) * epochs
    out: dict[str, Any] = {
        "rows": len(per_row), "epochs": epochs, "estimated_trained_tokens": trained,
        "rows_over_context": sum(t > ctx for t in per_row),
        "price_per_mtok": settings.price_per_mtok, "cap_usd": settings.max_cost_usd,
        "spent_usd": round(spent_usd(settings), 4),
        "avg_row_tokens": round(stats["avg_row_tokens"], 1),
        "spent_trained_tokens": spent_tokens(settings),
        "basis": f"{stats['basis']}; Nebius bills by compute, so treat as an estimate",
    }
    if settings.price_per_mtok is None:
        if settings.allow_unpriced:  # explicit opt-in: cap trained tokens instead of dollars
            out.update(estimated_cost_usd=None, max_trained_tokens=settings.max_trained_tokens,
                       within_budget=(out["spent_trained_tokens"] + trained
                                      <= settings.max_trained_tokens),
                       note="UNPRICED: no price configured, so launches are limited by a CUMULATIVE "
                            "trained-token cap (this job plus all earlier ones) instead of USD; "
                            "the dollar cost is unknown")
            return out
        out.update(estimated_cost_usd=None, within_budget=None,
                   note="set POSTTRAIN_PRICE_PER_MTOK to price the job (or opt in with "
                        "POSTTRAIN_ALLOW_UNPRICED=1); launch is blocked until then")
        return out
    cost = round(trained / 1e6 * settings.price_per_mtok, 4)
    out.update(estimated_cost_usd=cost,
               within_budget=out["spent_usd"] + cost <= settings.max_cost_usd)
    return out


def training_token_info(settings: Settings, dataset_ref: str) -> dict[str, Any]:
    """Per-epoch tokens and average row length as TRAINING sees them: the pretokenized file's
    figures when its check marker exists (those rows end at the tool call), else the full-row
    estimate. Returns {total_tokens, avg_row_tokens, rows, source} or {error}."""
    from .dataset import check_marker_path

    try:
        marker = json.loads(check_marker_path(Path(settings.data_dir), dataset_ref).read_text())
        if marker.get("ok") and marker.get("rows"):
            return {"total_tokens": marker["total_tokens"], "avg_row_tokens": marker["avg_row_tokens"],
                    "rows": marker["rows"], "source": "pretokenized"}
    except (OSError, ValueError, KeyError):
        pass
    stats = dataset_token_stats(settings, dataset_ref)
    if "error" in stats:
        return stats
    return {"total_tokens": stats["total_tokens"], "avg_row_tokens": stats["avg_row_tokens"],
            "rows": stats["rows"], "source": "estimate"}
