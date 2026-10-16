"""Run telemetry: a LangChain callback that logs model/tool events, plus a reader.

Events are appended to ``<state_dir>/telemetry.jsonl``. Prompts, completions and tool
arguments are NOT logged - only names, latencies, token counts and error types.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


class TelemetryHandler(BaseCallbackHandler):
    run_inline = True  # keep event order deterministic under asyncio

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._starts: dict[UUID, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def _emit(self, **row: Any) -> None:
        row["ts"] = time.time()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")

    def _start(self, run_id: UUID, name: str) -> None:
        self._starts[run_id] = (time.monotonic(), name)

    def _finish(self, run_id: UUID) -> tuple[float, str]:
        t0, name = self._starts.pop(run_id, (time.monotonic(), "?"))
        return round((time.monotonic() - t0) * 1000, 1), name

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kw):
        params = (metadata or {})
        name = f"{params.get('ls_provider', '')}:{params.get('ls_model_name', '')}"
        self._start(run_id, name)
        self._thread = params.get("thread_id")

    def on_llm_end(self, response, *, run_id, **kw):
        ms, name = self._finish(run_id)
        usage: dict[str, Any] = {}
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or usage
        self._emit(event="model", name=name, ms=ms, ok=True,
                   input_tokens=usage.get("input_tokens", 0),
                   output_tokens=usage.get("output_tokens", 0))

    def on_llm_error(self, error, *, run_id, **kw):
        ms, name = self._finish(run_id)
        self._emit(event="model", name=name, ms=ms, ok=False, error=type(error).__name__)

    def on_tool_start(self, serialized, input_str, *, run_id, **kw):
        self._start(run_id, (serialized or {}).get("name") or kw.get("name") or "tool")

    def on_tool_end(self, output, *, run_id, **kw):
        ms, name = self._finish(run_id)
        self._emit(event="tool", name=name, ms=ms, ok=True)

    def on_tool_error(self, error, *, run_id, **kw):
        ms, name = self._finish(run_id)
        self._emit(event="tool", name=name, ms=ms, ok=False, error=type(error).__name__)


def summarize(path: Path | str, last_n: int = 500, only_errors: bool = False) -> dict[str, Any]:
    """Aggregate the last ``last_n`` events: counts, error rate, latency, tokens."""
    path = Path(path)
    if not path.exists():
        return {"events": 0, "note": "no telemetry recorded yet"}
    rows = [json.loads(x) for x in path.read_text().splitlines()[-max(1, last_n):] if x.strip()]
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_name[f"{r['event']}:{r['name']}"].append(r)
    table = {}
    for key, items in sorted(by_name.items()):
        lat = sorted(i["ms"] for i in items)
        table[key] = {
            "calls": len(items), "errors": sum(not i["ok"] for i in items),
            "p50_ms": statistics.median(lat),
            "p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
            "input_tokens": sum(i.get("input_tokens", 0) for i in items),
            "output_tokens": sum(i.get("output_tokens", 0) for i in items),
        }
    errors = [r for r in rows if not r["ok"]]
    out: dict[str, Any] = {
        "events": len(rows), "window": last_n, "by_name": table,
        "error_types": dict(Counter(r.get("error") for r in errors)),
        "recent_errors": [{k: r[k] for k in ("ts", "event", "name", "error")} for r in errors[-5:]],
    }
    if only_errors:
        out.pop("by_name")
    return out


def telemetry_tool(path: Path | str):
    def read_telemetry(last_n: int = 500, only_errors: bool = False) -> dict[str, Any]:
        """Read run telemetry: model/tool call counts, error rates, latency p50/p95, token usage."""
        return summarize(path, last_n, only_errors)

    return read_telemetry
