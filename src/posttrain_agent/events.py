"""Redacting append-only event journal feeding the live dashboard.

Events are written to a SQLite file (WAL mode) and tailed by the read-only API. Only
allow-listed fields per event type are stored: prompts, completions, message text, tool
arguments and secrets never reach the journal. Writes are best-effort and never raise.
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
import re
import sqlite3
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CURRENT_RUN: contextvars.ContextVar[str] = ContextVar("CURRENT_RUN", default="")

MAX_STR = 300
MAX_DEPTH = 2
MAX_ITEMS = 32
_KEY_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,48}$")

_ALLOWED: dict[str, frozenset[str]] = {
    "run.started": frozenset({"goal_len", "experiment_budget", "backend"}),
    "run.status": frozenset({"status", "attempt", "errors"}),
    "agent.start": frozenset({"role", "slot"}),
    "agent.end": frozenset({"role", "slot", "ok"}),
    "model.call": frozenset(
        {"name", "ms", "input_tokens", "output_tokens", "ok", "error", "fallback"}
    ),
    "tool.call": frozenset({"name", "ms", "ok", "error"}),
    "score": frozenset({"split", "checkpoint", "value", "n", "evidence"}),
    "gate.verdict": frozenset(
        {"decision", "reasons", "champion_score", "candidate_score", "relative_improvement"}
    ),
    "budget.update": frozenset({"spent_usd", "cap_usd", "estimate_usd", "price_known"}),
    "job.update": frozenset(
        {"job_id", "status", "trained_steps", "total_steps", "trained_tokens", "evidence"}
    ),
    "job.log": frozenset({"level", "message"}),
    "approval.pending": frozenset(
        {"action", "base_model", "dataset_ref", "hyperparameters", "estimate_usd", "cli_hint"}
    ),
    "approval.resolved": frozenset({"decision", "message_len"}),
    "checkpoint": frozenset({"checkpoint_id", "step"}),
    "telemetry.snapshot": frozenset(
        {"p50_ms", "p95_ms", "error_rate", "input_tokens", "output_tokens"}
    ),
}
EVENT_TYPES: frozenset[str] = frozenset(_ALLOWED)

_DROP = object()


def _clean(value: Any, depth: int) -> Any:
    """Keep JSON scalars and lists/dicts of them up to MAX_DEPTH container levels."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:MAX_STR]
    if isinstance(value, (list, tuple)):
        if depth > MAX_DEPTH:
            return _DROP
        items = (_clean(v, depth + 1) for v in list(value)[:MAX_ITEMS])
        return [v for v in items if v is not _DROP]
    if isinstance(value, dict):
        if depth > MAX_DEPTH:
            return _DROP
        out: dict[str, Any] = {}
        for k, v in list(value.items())[:MAX_ITEMS]:
            if not isinstance(k, str) or not _KEY_RE.match(k):
                continue
            cleaned = _clean(v, depth + 1)
            if cleaned is not _DROP:
                out[k] = cleaned
        return out
    return _DROP


def sanitize(type: str, data: dict) -> dict | None:
    """Return only the allow-listed, size-bounded fields, or None for an unknown type."""
    allowed = _ALLOWED.get(type)
    if allowed is None:
        return None
    out: dict[str, Any] = {}
    for key, value in (data or {}).items():
        if key not in allowed:
            continue
        cleaned = _clean(value, 1)
        if cleaned is not _DROP:
            out[key] = cleaned
    return out


@dataclass
class Event:
    seq: int
    ts: float
    run: str
    type: str
    data: dict


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    run TEXT NOT NULL,
    type TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_run_seq ON events(run, seq);
"""


class Journal:
    def __init__(
        self, path: Path | str, max_rows: int = 100_000, prune_every: int = 1_000
    ) -> None:
        self.path = Path(path)
        self.max_rows = max_rows
        self.prune_every = prune_every
        self._writes = 0
        self._warned = False
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect():
                pass
        except Exception as exc:  # noqa: BLE001 - journal must never fail a run
            self._warn(exc)

    def _warn(self, exc: BaseException) -> None:
        if not self._warned:
            self._warned = True
            log.warning("event journal unavailable (%s: %s)", type(exc).__name__, exc)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=1)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
        except Exception:
            conn.close()
            raise
        return conn

    def append(self, run: str, type: str, data: dict) -> int | None:
        try:
            clean = sanitize(type, data)
            if clean is None:
                return None
            payload = json.dumps(clean, separators=(",", ":"))
            with self._lock:
                self._writes += 1
                prune = self.prune_every > 0 and self._writes % self.prune_every == 0
            conn = self._connect()
            try:
                with conn:
                    cur = conn.execute(
                        "INSERT INTO events(ts, run, type, data) VALUES (?, ?, ?, ?)",
                        (time.time(), run, type, payload),
                    )
                    seq = cur.lastrowid
                    if prune:
                        conn.execute("DELETE FROM events WHERE seq <= ?", (seq - self.max_rows,))
            finally:
                conn.close()
            return seq
        except Exception as exc:  # noqa: BLE001
            self._warn(exc)
            return None

    def read(self, run: str | None = None, since: int = 0, limit: int = 500) -> list[Event]:
        sql = "SELECT seq, ts, run, type, data FROM events WHERE seq > ?"
        args: list[Any] = [since]
        if run is not None:
            sql += " AND run = ?"
            args.append(run)
        sql += " ORDER BY seq ASC LIMIT ?"
        args.append(limit)
        try:
            conn = self._connect()
            try:
                rows = conn.execute(sql, args).fetchall()
            finally:
                conn.close()
            return [Event(s, t, r, ty, json.loads(d)) for s, t, r, ty, d in rows]
        except Exception as exc:  # noqa: BLE001
            self._warn(exc)
            return []

    def runs(self) -> list[dict]:
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT run, MIN(ts), MAX(seq) FROM events GROUP BY run ORDER BY MAX(seq)"
                ).fetchall()
                out = []
                for run, started, last_seq in rows:
                    row = conn.execute(
                        "SELECT data FROM events WHERE run = ? AND type = 'run.status' "
                        "ORDER BY seq DESC LIMIT 1",
                        (run,),
                    ).fetchone()
                    status = json.loads(row[0]).get("status") if row else None
                    out.append(
                        {"run": run, "started": started, "last_seq": last_seq, "status": status}
                    )
                return out
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            self._warn(exc)
            return []

    def last_seq(self) -> int:
        try:
            conn = self._connect()
            try:
                row = conn.execute("SELECT MAX(seq) FROM events").fetchone()
            finally:
                conn.close()
            return int(row[0] or 0)
        except Exception as exc:  # noqa: BLE001
            self._warn(exc)
            return 0
