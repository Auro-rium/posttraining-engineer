"""Read-only FastAPI service streaming the event journal to the live dashboard over SSE.

The API never appends to the journal. With ``replay_file`` it serves a recorded ``.jsonl``
journal (one event per line: seq, ts, run, type, data) through the same routes, paced by the
original timestamps divided by ``?speed=``.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import sqlite3
import sys
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import load_dotenv
from .observability import fold
from .observability.events import Event

MAX_REPLAY_SLEEP_S = 2.0
BATCH = 500


def _load_replay(path: Path) -> list[Event]:
    events: list[Event] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            events.append(
                Event(int(d["seq"]), float(d["ts"]), str(d["run"]), str(d["type"]), dict(d["data"]))
            )
        except (ValueError, KeyError, TypeError):
            continue
    events.sort(key=lambda e: e.seq)
    return events


def _runs_of(events: list[Event]) -> list[dict]:
    by_run: dict[str, dict] = {}
    for e in events:
        r = by_run.setdefault(
            e.run, {"run": e.run, "started": e.ts, "last_seq": e.seq, "status": None}
        )
        r["started"] = min(r["started"], e.ts)
        r["last_seq"] = max(r["last_seq"], e.seq)
        if e.type == "run.status":
            r["status"] = e.data.get("status")
    return sorted(by_run.values(), key=lambda r: r["last_seq"])


def _frame(e: Event) -> str:
    return f"id: {e.seq}\nevent: {e.type}\ndata: {json.dumps(asdict(e), separators=(',', ':'))}\n\n"


class ReadOnlyJournal:
    """Reads the SQLite journal with ``mode=ro``: never creates files and never runs DDL."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=1)

    def probe(self) -> bool:
        try:
            conn = self._connect()
            try:
                conn.execute("SELECT 1 FROM events LIMIT 1").fetchall()
            finally:
                conn.close()
            return True
        except Exception:  # noqa: BLE001
            return False

    def read(self, run: str | None, since: int, limit: int) -> list[Event]:
        sql = "SELECT seq, ts, run, type, data FROM events WHERE seq > ?"
        args: list = [since]
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
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            return []


class _SlotResponse(StreamingResponse):
    """StreamingResponse that always releases its stream slot, even if never started."""

    def __init__(self, *args, release: Callable[[], None], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._release = release

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._release()


def create_app(
    state_dir: Path,
    read_token: str,
    allowed_origin: str,
    replay_file: Path | None = None,
    max_streams: int = 20,
    poll_s: float = 0.25,
    heartbeat_s: float = 15.0,
) -> FastAPI:
    app = FastAPI(title="posttrain live API", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[allowed_origin],
        allow_methods=["GET"],
        allow_headers=["Last-Event-ID", "Content-Type"],
    )
    replay: list[Event] | None = _load_replay(Path(replay_file)) if replay_file else None
    journal = None if replay is not None else ReadOnlyJournal(Path(state_dir) / "events.sqlite")
    active = {"n": 0}
    app.state.active = active

    def journal_ok() -> bool:
        return replay is not None or journal.probe()  # type: ignore[union-attr]

    def require_journal() -> None:
        if not journal_ok():
            raise HTTPException(status_code=503, detail="journal unavailable")

    def auth(token: str = Query(default="")) -> None:
        if not read_token or not hmac.compare_digest(token.encode(), read_token.encode()):
            raise HTTPException(status_code=401, detail="invalid token")

    def read(run: str | None, since: int) -> list[Event]:
        if replay is not None:
            out = [e for e in replay if e.seq > since and (run is None or e.run == run)]
            return out[:BATCH]
        assert journal is not None
        return journal.read(run=run, since=since, limit=BATCH)

    async def idle_tail(request: Request, last_sent: list[float]) -> AsyncIterator[str]:
        while not await request.is_disconnected():
            await asyncio.sleep(min(poll_s, 0.5))
            if time.monotonic() - last_sent[0] >= heartbeat_s:
                last_sent[0] = time.monotonic()
                yield ": hb\n\n"

    async def live_events(request: Request, run: str | None, since: int) -> AsyncIterator[str]:
        last = since
        last_sent = [time.monotonic()]
        while not await request.is_disconnected():
            events = await asyncio.to_thread(read, run, last)
            for e in events:
                last = e.seq
                last_sent[0] = time.monotonic()
                yield _frame(e)
            if events:
                continue
            await asyncio.sleep(poll_s)
            if time.monotonic() - last_sent[0] >= heartbeat_s:
                last_sent[0] = time.monotonic()
                yield ": hb\n\n"

    async def replay_events(
        request: Request, run: str | None, since: int, speed: float
    ) -> AsyncIterator[str]:
        last_sent = [time.monotonic()]
        prev_ts: float | None = None
        for e in await asyncio.to_thread(read_all, run, since):
            if await request.is_disconnected():
                return
            if prev_ts is not None:
                wait = min(max(e.ts - prev_ts, 0.0) / speed, MAX_REPLAY_SLEEP_S)
                if wait > 0:
                    await asyncio.sleep(wait)
            prev_ts = e.ts
            last_sent[0] = time.monotonic()
            yield _frame(e)
        async for hb in idle_tail(request, last_sent):
            yield hb

    def read_all(run: str | None, since: int) -> list[Event]:
        assert replay is not None
        return [e for e in replay if e.seq > since and (run is None or e.run == run)]

    @app.get("/v1/health")
    def health() -> JSONResponse:
        mode = "replay" if replay is not None else "live"
        if journal_ok():
            return JSONResponse({"ok": True, "mode": mode, "journal": True})
        return JSONResponse(
            {"ok": False, "status": "degraded", "mode": mode, "journal": False}, status_code=503
        )

    @app.get("/v1/runs", dependencies=[Depends(auth)])
    def runs() -> dict[str, Any]:
        require_journal()
        return {"runs": _runs_of(replay) if replay is not None else journal.runs()}  # type: ignore[union-attr]

    def run_events(run_id: str) -> list[Event]:
        events: list[Event] = []
        since = 0
        while True:
            batch = (
                [e for e in replay if e.seq > since and e.run == run_id][:BATCH]
                if replay is not None
                else journal.read(run=run_id, since=since, limit=BATCH)  # type: ignore[union-attr]
            )
            if not batch:
                return events
            events.extend(batch)
            since = batch[-1].seq

    @app.get("/v1/runs/{run_id}/snapshot", dependencies=[Depends(auth)])
    def snapshot(run_id: str) -> dict[str, Any]:
        require_journal()
        state = fold.snapshot(run_events(run_id))
        return {"state": state, "last_seq": state.get("last_seq", 0)}

    @app.get("/v1/runs/{run_id}/a2a", dependencies=[Depends(auth)])
    def a2a(run_id: str) -> dict[str, Any]:
        """The full agent-to-agent conversation of a run (the snapshot keeps only the last 100)."""
        require_journal()
        return {"run": run_id, "messages": fold.a2a_conversation(run_events(run_id))}

    @app.post("/v1/runs/{run_id}/decision", dependencies=[Depends(auth)])
    def decision(run_id: str) -> None:
        raise HTTPException(status_code=501, detail="not implemented")

    @app.get("/v1/stream", dependencies=[Depends(auth)])
    async def stream(
        request: Request,
        run: str | None = None,
        since: int = 0,
        speed: float = Query(default=1.0, gt=0),
    ) -> StreamingResponse:
        header = request.headers.get("last-event-id")
        if header is not None:
            try:
                since = max(int(header), 0)
            except ValueError:
                pass
        require_journal()
        if active["n"] >= max_streams:
            raise HTTPException(status_code=429, detail="too many streams")
        active["n"] += 1
        inner = (
            replay_events(request, run, since, speed)
            if replay is not None
            else live_events(request, run, since)
        )

        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                active["n"] -= 1

        return _SlotResponse(
            inner,
            release=release,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def main() -> None:
    load_dotenv()  # ./.env fills in anything not already exported (entry point only)
    token = os.environ.get("READ_TOKEN", "")
    if not token:
        print(
            "READ_TOKEN is empty; refusing to start the API without a read token.", file=sys.stderr
        )
        raise SystemExit(2)
    replay = os.environ.get("REPLAY_FILE") or None
    app = create_app(
        Path(os.environ.get("STATE_DIR") or "state"),
        token,
        os.environ.get("ALLOWED_ORIGIN", ""),
        replay_file=Path(replay) if replay else None,
    )
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT") or 8787), access_log=False)


if __name__ == "__main__":
    main()
