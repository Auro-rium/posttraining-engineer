"""LangChain callback that feeds the event journal (and telemetry.jsonl when given a path).

Only names, latencies, token counts and error types are journaled; the journal's allow-list
would drop anything else, and this handler never reads prompts, completions or tool args.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .events import CURRENT_RUN, Journal
from .telemetry import TelemetryHandler

log = logging.getLogger(__name__)

SNAPSHOT_EVERY_S = 5.0


class JournalHandler(TelemetryHandler):
    def __init__(
        self,
        journal: Journal,
        run: str = "",
        primary: str = "",
        telemetry_path: Path | str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(telemetry_path or "telemetry.jsonl")
        self._write_file = telemetry_path is not None
        self.journal = journal
        self.run = run
        self.primary = primary
        self._clock = clock
        self._last_snapshot = clock()
        self._recent: list[tuple[float, bool, int, int]] = []

    def _emit(self, **row: Any) -> None:
        if self._write_file:
            super()._emit(**dict(row))
        try:
            self._journal(row)
        except Exception as exc:  # noqa: BLE001 - journaling must never fail a run
            log.debug("journal emit failed: %s", exc)

    def _journal(self, row: dict[str, Any]) -> None:
        run = self.run or CURRENT_RUN.get()
        if row.get("event") == "tool":
            data = {k: row[k] for k in ("name", "ms", "ok", "error") if k in row}
            self.journal.append(run, "tool.call", data)
            return
        data = {k: row[k] for k in
                ("name", "ms", "ok", "error", "input_tokens", "output_tokens") if k in row}
        data["fallback"] = self.primary != "" and row.get("name") != self.primary
        self.journal.append(run, "model.call", data)
        self._recent.append((row.get("ms", 0.0), bool(row.get("ok")),
                             row.get("input_tokens", 0), row.get("output_tokens", 0)))
        if self._clock() - self._last_snapshot >= SNAPSHOT_EVERY_S:
            self._snapshot(run)

    def _snapshot(self, run: str) -> None:
        calls, self._recent = self._recent, []
        self._last_snapshot = self._clock()
        lat = sorted(c[0] for c in calls)
        self.journal.append(run, "telemetry.snapshot", {
            "p50_ms": statistics.median(lat),
            "p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
            "error_rate": sum(not c[1] for c in calls) / len(calls),
            "input_tokens": sum(c[2] for c in calls),
            "output_tokens": sum(c[3] for c in calls),
        })
