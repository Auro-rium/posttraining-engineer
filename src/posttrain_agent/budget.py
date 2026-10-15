"""Deterministic training-cost estimate and cumulative spend ledger.

Nebius Token Factory publishes no per-token fine-tuning price, so the price is
operator-supplied (``POSTTRAIN_PRICE_PER_MTOK``). Without it the estimate reports
token counts only and ``within_budget`` is ``None``; launching then fails closed.
Token counts are a chars/4 heuristic; actual billing is by compute and may differ.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import Settings
from .dataset import load_rows


def _row_tokens(row: dict[str, Any]) -> int:
    chars = 0
    for m in row.get("messages", []):
        chars += len(m.get("content") or "") + 16  # role/format overhead
        for c in m.get("tool_calls") or []:
            fn = c.get("function", {})
            chars += len(fn.get("name", "")) + len(str(fn.get("arguments", "")))
    return max(1, chars // 4)


def spend_path(settings: Settings) -> Path:
    return Path(settings.state_dir) / "spend.jsonl"


def spent_usd(settings: Settings) -> float:
    path = spend_path(settings)
    if not path.exists():
        return 0.0
    return sum(json.loads(line)["estimated_cost_usd"] for line in path.read_text().splitlines()
               if line.strip())


def record_spend(settings: Settings, job_id: str | None, cost: float, dataset_ref: str) -> None:
    path = spend_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps({"ts": time.time(), "job_id": job_id, "dataset_ref": dataset_ref,
                             "estimated_cost_usd": cost}) + "\n")


def estimate(settings: Settings, dataset_ref: str, hyperparameters: dict[str, Any]) -> dict[str, Any]:
    try:
        rows = load_rows(settings.data_dir, dataset_ref)
    except (OSError, ValueError) as exc:
        return {"error": f"cannot read dataset: {exc}"}
    epochs = hyperparameters.get("n_epochs", 3)
    ctx = hyperparameters.get("context_length", 8192)
    per_row = [_row_tokens(r) for r in rows if isinstance(r, dict)]
    trained = sum(min(t, ctx) for t in per_row) * epochs
    out: dict[str, Any] = {
        "rows": len(per_row), "epochs": epochs, "estimated_trained_tokens": trained,
        "rows_over_context": sum(t > ctx for t in per_row),
        "price_per_mtok": settings.price_per_mtok, "cap_usd": settings.max_cost_usd,
        "spent_usd": round(spent_usd(settings), 4),
        "basis": "chars/4 heuristic; Nebius bills by compute, so treat as an estimate",
    }
    if settings.price_per_mtok is None:
        out.update(estimated_cost_usd=None, within_budget=None,
                   note="set POSTTRAIN_PRICE_PER_MTOK to price the job; launch is blocked until set")
        return out
    cost = round(trained / 1e6 * settings.price_per_mtok, 4)
    out.update(estimated_cost_usd=cost,
               within_budget=out["spent_usd"] + cost <= settings.max_cost_usd)
    return out
