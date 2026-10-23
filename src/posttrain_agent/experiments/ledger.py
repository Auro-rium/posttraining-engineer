"""Append-only evaluation ledger: ``<state_dir>/eval_ledger.jsonl``, one JSON object per line.

Record (written by ``domain.make_tools.evaluate_checkpoint`` after every evaluation attempt)::

    {"ts": ISO-8601 UTC, "run": thread/run id or null, "checkpoint": resolved checkpoint name,
     "job": "ftjob-..." when the caller passed a job id else null, "split": "heldout"|"regression",
     "suite": ..., "seed": ..., "status": "ok" | "error", "result": <evaluate() dict or error dict>}

``status == "ok"`` means ``result`` holds a ``score``; ``"error"`` means a structured error
(``error``, ``stage``, ``transient``) and never a score. Readers decide pairing: a candidate is
usable only when BOTH splits have an ok record.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LEDGER_NAME = "eval_ledger.jsonl"
_LOCK = threading.Lock()


def ledger_path(state_dir: Path | str) -> Path:
    return Path(state_dir) / LEDGER_NAME


def append(state_dir: Path | str, *, run: str | None, checkpoint: str, split: str,
           result: dict[str, Any], job: str | None = None, suite: str | None = None,
           seed: int | None = None) -> dict[str, Any]:
    rec = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "run": run,
           "checkpoint": checkpoint, "job": job, "split": split,
           "suite": suite if suite is not None else result.get("suite"),
           "seed": seed if seed is not None else result.get("seed"),
           "status": "ok" if "score" in result and "error" not in result else "error",
           "result": result}
    path = ledger_path(state_dir)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def read(state_dir: Path | str) -> list[dict[str, Any]]:
    path = ledger_path(state_dir)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # a torn last line must not hide the rest
        if isinstance(rec, dict):
            out.append(rec)
    return out


def latest_ok(state_dir: Path | str, *, checkpoint: str | None = None, job: str | None = None,
              split: str, run: str | None = None, suite: str | None = None,
              seed: int | None = None) -> dict[str, Any] | None:
    """The newest ok record matching every given filter (None filters are ignored)."""
    for rec in reversed(read(state_dir)):
        if rec.get("status") != "ok" or rec.get("split") != split:
            continue
        if any(want is not None and rec.get(k) != want for k, want in
               (("checkpoint", checkpoint), ("job", job), ("run", run), ("suite", suite),
                ("seed", seed))):
            continue
        return rec
    return None


def latest_matching(state_dir: Path | str, *, checkpoint: str, split: str,
                    like: dict[str, Any]) -> dict[str, Any] | None:
    """Newest ok record for ``checkpoint``/``split`` computed on the same task set, judge and
    serving path as ``like`` (an evaluate() result), so a stored base result is only reused when
    the promotion gate would accept the pairing."""
    for rec in reversed(read(state_dir)):
        res = rec.get("result") or {}
        if (rec.get("status") == "ok" and rec.get("checkpoint") == checkpoint
                and rec.get("split") == split
                and all(res.get(k) == like.get(k) for k in
                        ("suite", "seed", "judged_by", "served_by", "tasks_sha256"))):
            return rec
    return None
