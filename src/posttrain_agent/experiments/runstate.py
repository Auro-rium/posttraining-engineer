"""Durable run-state, launch-intent, gate-verdict, abort and operator-note records.

All files live in ``state_dir`` as append-only JSONL, latest record wins per key:

* ``run_state.jsonl``  one record per training job (and per launch intent before a job id exists)::
    {ts, run, key, job_id|null, dataset_ref, dataset_sha256, hparams_hash, hparams,
     status, checkpoint, trained_tokens}
  ``key`` = sha256(dataset sha256 + hparams hash): a second launch with the same key is refused
  unless the earlier job is terminal-failed.
* ``gate_log.jsonl``   one record per run_gate verdict {ts, run, champion, candidate, candidate_job, decision}
* ``aborts.jsonl``     one record per abort_experiment {ts, run, reason}
* ``notes.jsonl``      operator notes {ts, thread, text, id}; ``notes_read.json`` holds the per-thread cursor

Nothing here calls a model or the network.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
TERMINAL_FAILED = frozenset({"failed", "cancelled", "canceled", "launch_failed", "blocked"})
SUCCEEDED = "succeeded"
TERMINAL = TERMINAL_FAILED | {SUCCEEDED}


def hparams_hash(hparams: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(hparams, sort_keys=True, default=str).encode()).hexdigest()[:16]


def intent_key(dataset_sha256: str, hparams: dict[str, Any]) -> str:
    return hashlib.sha256(f"{dataset_sha256}|{hparams_hash(hparams)}".encode()).hexdigest()[:24]


def _append(path: Path, rec: dict[str, Any]) -> dict[str, Any]:
    rec = {"ts": time.time(), **rec}
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    return rec


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


# ---- run state ----------------------------------------------------------------------------

def _rs(state_dir: Path | str) -> Path:
    return Path(state_dir) / "run_state.jsonl"


def write_run_state(state_dir: Path | str, **rec: Any) -> dict[str, Any]:
    return _append(_rs(state_dir), rec)


def run_states(state_dir: Path | str, run: str | None = None) -> list[dict[str, Any]]:
    """Latest merged record per (key, job) in first-seen order; optionally one run only."""
    merged: dict[str, dict[str, Any]] = {}
    for rec in _read(_rs(state_dir)):
        if run and rec.get("run") != run:
            continue
        ident = rec.get("job_id") or rec.get("key") or ""
        # an intent record (no job id yet) is superseded by the same key's job record
        if rec.get("job_id") and rec.get("key") in merged:
            merged[rec["job_id"]] = {**merged.pop(rec["key"]), **rec}
            continue
        merged[ident] = {**merged.get(ident, {}), **rec}
    firsts: dict[str, float] = {}  # earliest record per job/key: when the candidate was launched
    for rec in _read(_rs(state_dir)):
        for k in (rec.get("key"), rec.get("job_id")):
            if k:
                firsts.setdefault(k, float(rec.get("ts", 0)))
    for rec in merged.values():
        ts = [firsts[k] for k in (rec.get("key"), rec.get("job_id")) if k in firsts]
        if ts:
            rec["first_ts"] = min(ts)
    return list(merged.values())


def find_by_key(state_dir: Path | str, key: str) -> dict[str, Any] | None:
    found = [r for r in run_states(state_dir) if r.get("key") == key]
    return found[-1] if found else None


def find_job(state_dir: Path | str, job_id: str) -> dict[str, Any] | None:
    for r in reversed(run_states(state_dir)):
        if r.get("job_id") == job_id:
            return r
    return None


def current_candidate(state_dir: Path | str, run: str | None) -> dict[str, Any] | None:
    """Newest run-state record of ``run`` (or of any run when ``run`` is empty)."""
    rows = run_states(state_dir, run or None)
    return rows[-1] if rows else None


def latest_succeeded(state_dir: Path | str, run: str | None) -> dict[str, Any] | None:
    for r in reversed(run_states(state_dir, run or None)):
        if str(r.get("status", "")).lower() == SUCCEEDED and r.get("checkpoint"):
            return r
    return None


# ---- gate log / aborts ----------------------------------------------------------------------

def record_gate(state_dir: Path | str, **rec: Any) -> dict[str, Any]:
    return _append(Path(state_dir) / "gate_log.jsonl", rec)


def gate_records(state_dir: Path | str, run: str) -> list[dict[str, Any]]:
    return [r for r in _read(Path(state_dir) / "gate_log.jsonl") if r.get("run") == run]


def record_abort(state_dir: Path | str, run: str, reason: str) -> dict[str, Any]:
    return _append(Path(state_dir) / "aborts.jsonl", {"run": run, "reason": reason})


def abort_records(state_dir: Path | str, run: str) -> list[dict[str, Any]]:
    return [r for r in _read(Path(state_dir) / "aborts.jsonl") if r.get("run") == run]


# ---- operator notes -------------------------------------------------------------------------

def add_note(state_dir: Path | str, thread: str, text: str) -> dict[str, Any]:
    n = len(_read(Path(state_dir) / "notes.jsonl")) + 1
    return _append(Path(state_dir) / "notes.jsonl", {"thread": thread, "text": text, "id": n})


def unread_notes(state_dir: Path | str, thread: str, *, mark: bool = True) -> list[dict[str, Any]]:
    """Notes for ``thread`` not yet delivered; advances the cursor so each is delivered once."""
    cursor_path = Path(state_dir) / "notes_read.json"
    with _LOCK:
        try:
            cursors = json.loads(cursor_path.read_text())
        except (OSError, ValueError):
            cursors = {}
        seen = int(cursors.get(thread, 0))
        new = [r for r in _read(Path(state_dir) / "notes.jsonl")
               if r.get("thread") == thread and int(r.get("id", 0)) > seen]
        if new and mark:
            cursors[thread] = max(int(r["id"]) for r in new)
            cursor_path.parent.mkdir(parents=True, exist_ok=True)
            cursor_path.write_text(json.dumps(cursors))
    return new
