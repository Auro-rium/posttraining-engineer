from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import httpx
import pytest

from posttrain_agent import api, fold
from posttrain_agent.events import Event, Journal

TOKEN = "s3cret"
ORIGIN = "https://dash.example.com"


def make_app(tmp_path: Path, **kw):
    kw.setdefault("poll_s", 0.01)
    kw.setdefault("heartbeat_s", 15.0)
    return api.create_app(tmp_path, TOKEN, ORIGIN, **kw)


def seed(tmp_path: Path, n: int = 5, run: str = "r1") -> Journal:
    j = Journal(tmp_path / "events.sqlite")
    for i in range(n):
        j.append(run, "run.status", {"status": "running", "attempt": i})
    return j


def client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def raw_stream(app, path, headers=None, stop=None, timeout=5.0):
    """Drive the ASGI app directly, collecting body text until `stop(text)` then disconnect."""
    hdrs = [(b"host", b"t")] + [
        (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
    ]
    path_only, _, qs = path.partition("?")
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
        "scheme": "http", "path": path_only, "raw_path": path_only.encode(),
        "query_string": qs.encode(), "headers": hdrs, "server": ("t", 80), "client": ("c", 1),
    }  # fmt: skip
    disconnect = asyncio.Event()
    state = {"status": None, "headers": {}, "text": ""}

    async def receive():
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(msg):
        if msg["type"] == "http.response.start":
            state["status"] = msg["status"]
            state["headers"] = {k.decode(): v.decode() for k, v in msg["headers"]}
        elif msg["type"] == "http.response.body":
            state["text"] += msg.get("body", b"").decode()
            if stop is None or stop(state["text"]):
                disconnect.set()

    if stop is None:
        disconnect.set()
    await asyncio.wait_for(app(scope, receive, send), timeout)
    return state


def parse_frames(text: str) -> list[dict]:
    frames = []
    for block in text.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        f = {}
        for line in block.split("\n"):
            k, _, v = line.partition(": ")
            f[k] = v
        frames.append(f)
    return frames


def ids(text: str) -> list[int]:
    return [int(f["id"]) for f in parse_frames(text)]


async def test_missing_or_wrong_token_is_401(tmp_path):
    seed(tmp_path, 1)
    app = make_app(tmp_path)
    async with client(app) as c:
        assert (await c.get("/v1/runs")).status_code == 401
        assert (await c.get("/v1/runs?token=nope")).status_code == 401
        assert (await c.get("/v1/runs/x/snapshot")).status_code == 401
        assert (await c.get(f"/v1/runs?token={TOKEN}")).status_code == 200
    st = await raw_stream(app, "/v1/stream?token=bad")
    assert st["status"] == 401


async def test_cors_only_allows_configured_origin(tmp_path):
    app = make_app(tmp_path)
    async with client(app) as c:
        ok = await c.get("/v1/health", headers={"Origin": ORIGIN})
        assert ok.headers["access-control-allow-origin"] == ORIGIN
        bad = await c.get("/v1/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in bad.headers
        pre = await c.options(
            "/v1/runs",
            headers={"Origin": ORIGIN, "Access-Control-Request-Method": "GET"},
        )
        assert pre.status_code == 200
        assert pre.headers["access-control-allow-origin"] == ORIGIN
        pre_bad = await c.options(
            "/v1/runs",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
        )
        assert "access-control-allow-origin" not in pre_bad.headers
        err = await c.get("/v1/runs", headers={"Origin": ORIGIN})
        assert err.status_code == 401
        assert err.headers["access-control-allow-origin"] == ORIGIN


async def test_stream_emits_frames_with_seq_ids_and_event_names(tmp_path):
    seed(tmp_path, 3)
    app = make_app(tmp_path)
    st = await raw_stream(app, f"/v1/stream?token={TOKEN}", stop=lambda t: t.count("\n\n") >= 3)
    assert st["status"] == 200
    assert st["headers"]["content-type"].startswith("text/event-stream")
    assert st["text"].startswith("id: 1\nevent: run.status\ndata: ")
    frames = parse_frames(st["text"])
    assert [f["id"] for f in frames] == ["1", "2", "3"]
    assert json.loads(frames[1]["data"])["data"] == {"status": "running", "attempt": 1}


async def test_resume_with_last_event_id_has_no_gaps_or_duplicates(tmp_path):
    j = seed(tmp_path, 5)
    app = make_app(tmp_path)
    st = await raw_stream(
        app,
        f"/v1/stream?token={TOKEN}",
        headers={"Last-Event-ID": "2"},
        stop=lambda t: t.count("\n\n") >= 3,
    )
    assert ids(st["text"]) == [3, 4, 5]
    j.append("r1", "run.status", {"status": "done"})
    st2 = await raw_stream(
        app,
        f"/v1/stream?token={TOKEN}",
        headers={"Last-Event-ID": "5"},
        stop=lambda t: t.count("\n\n") >= 1,
    )
    assert ids(st2["text"]) == [6]


async def test_last_event_id_beats_since_param(tmp_path):
    seed(tmp_path, 5)
    app = make_app(tmp_path)
    st = await raw_stream(
        app,
        f"/v1/stream?token={TOKEN}&since=4",
        headers={"Last-Event-ID": "1"},
        stop=lambda t: t.count("\n\n") >= 4,
    )
    assert ids(st["text"]) == [2, 3, 4, 5]
    st = await raw_stream(
        app, f"/v1/stream?token={TOKEN}&since=3", stop=lambda t: t.count("\n\n") >= 2
    )
    assert ids(st["text"]) == [4, 5]


async def test_heartbeat_comment_sent_when_idle(tmp_path):
    seed(tmp_path, 0)
    app = make_app(tmp_path, heartbeat_s=0.05)
    st = await raw_stream(app, f"/v1/stream?token={TOKEN}", stop=lambda t: ": hb\n\n" in t)
    assert ": hb\n\n" in st["text"]


async def test_run_filter_isolates_interleaved_runs(tmp_path):
    j = Journal(tmp_path / "events.sqlite")
    for i in range(3):
        j.append("a", "run.status", {"status": "running", "attempt": i})
        j.append("b", "checkpoint", {"checkpoint_id": f"c{i}", "step": i})
    app = make_app(tmp_path)
    st = await raw_stream(
        app, f"/v1/stream?token={TOKEN}&run=b", stop=lambda t: t.count("\n\n") >= 3
    )
    frames = parse_frames(st["text"])
    assert [f["event"] for f in frames] == ["checkpoint"] * 3
    assert [f["id"] for f in frames] == ["2", "4", "6"]
    async with client(app) as c:
        runs = (await c.get(f"/v1/runs?token={TOKEN}")).json()
    assert {r["run"] for r in runs["runs"]} == {"a", "b"}


async def test_snapshot_equals_fold_of_stream(tmp_path):
    j = Journal(tmp_path / "events.sqlite")
    j.append("r", "run.started", {"backend": "x", "goal_len": 3, "experiment_budget": 2})
    j.append("other", "run.status", {"status": "noise"})
    j.append("r", "run.status", {"status": "running", "attempt": 1})
    j.append("r", "agent.start", {"role": "trainer", "slot": 0})
    app = make_app(tmp_path)
    st = await raw_stream(
        app, f"/v1/stream?token={TOKEN}&run=r", stop=lambda t: t.count("\n\n") >= 3
    )
    evs = [Event(**json.loads(f["data"])) for f in parse_frames(st["text"])]
    async with client(app) as c:
        body = (await c.get(f"/v1/runs/r/snapshot?token={TOKEN}")).json()
    assert body["state"] == fold.snapshot(evs)
    assert body["last_seq"] == 4


async def test_reader_survives_concurrent_writer(tmp_path):
    j = Journal(tmp_path / "events.sqlite")
    n = 40

    def writer():
        for i in range(n):
            j.append("w", "run.status", {"status": "running", "attempt": i})
            time.sleep(0.002)

    t = threading.Thread(target=writer)
    app = make_app(tmp_path)
    t.start()
    try:
        st = await raw_stream(
            app, f"/v1/stream?token={TOKEN}", stop=lambda x: x.count("\n\n") >= n, timeout=15
        )
    finally:
        t.join()
    assert ids(st["text"]) == list(range(1, n + 1))


async def test_21st_stream_gets_429(tmp_path):
    seed(tmp_path, 1)
    app = make_app(tmp_path, max_streams=2)
    gate = asyncio.Event()
    tasks = [
        asyncio.create_task(
            raw_stream(app, f"/v1/stream?token={TOKEN}", stop=lambda t: gate.is_set() and True)
        )
        for _ in range(2)
    ]
    await asyncio.sleep(0.3)
    st = await raw_stream(app, f"/v1/stream?token={TOKEN}")
    assert st["status"] == 429
    gate.set()
    # Third stream's frame arrival makes the held ones stop on their next body message.
    j = Journal(tmp_path / "events.sqlite")
    j.append("r1", "run.status", {"status": "x"})
    await asyncio.wait_for(asyncio.gather(*tasks), 5)
    again = await raw_stream(app, f"/v1/stream?token={TOKEN}", stop=lambda t: "\n\n" in t)
    assert again["status"] == 200


async def test_decision_endpoint_is_501(tmp_path):
    app = make_app(tmp_path)
    async with client(app) as c:
        r = await c.post(f"/v1/runs/r/decision?token={TOKEN}", json={"decision": "approve"})
    assert r.status_code == 501


def test_main_refuses_to_start_without_read_token(monkeypatch, capsys):
    monkeypatch.delenv("READ_TOKEN", raising=False)
    with pytest.raises(SystemExit) as ei:
        api.main()
    assert ei.value.code not in (0, None)
    assert "READ_TOKEN" in capsys.readouterr().err
    monkeypatch.setenv("READ_TOKEN", "")
    with pytest.raises(SystemExit):
        api.main()


async def test_replay_mode_serves_recorded_journal(tmp_path):
    f = tmp_path / "rec.jsonl"
    rows = [
        {"seq": 1, "ts": 100.0, "run": "x", "type": "run.status", "data": {"status": "running"}},
        {"seq": 2, "ts": 100.2, "run": "x", "type": "checkpoint",
         "data": {"checkpoint_id": "c", "step": 1}},
        {"seq": 3, "ts": 400.0, "run": "x", "type": "run.status", "data": {"status": "done"}},
    ]  # fmt: skip
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    app = make_app(tmp_path / "state", replay_file=f)
    async with client(app) as c:
        assert (await c.get(f"/v1/runs?token={TOKEN}")).json()["runs"][0]["run"] == "x"
        snap = (await c.get(f"/v1/runs/x/snapshot?token={TOKEN}")).json()
        assert snap["last_seq"] == 3 and snap["state"]["status"] == "done"
    t0 = time.monotonic()
    st = await raw_stream(
        app, f"/v1/stream?token={TOKEN}&speed=1000", stop=lambda t: t.count("\n\n") >= 3
    )
    assert ids(st["text"]) == [1, 2, 3]
    assert time.monotonic() - t0 < 2.5
    assert not (tmp_path / "state" / "events.sqlite").exists()


async def test_stream_data_is_full_envelope(tmp_path):
    j = seed(tmp_path, 1)
    app = make_app(tmp_path)
    st = await raw_stream(app, f"/v1/stream?token={TOKEN}", stop=lambda t: "\n\n" in t)
    env = json.loads(parse_frames(st["text"])[0]["data"])
    assert set(env) == {"seq", "ts", "run", "type", "data"}
    assert env["ts"] == j.read()[0].ts and env["run"] == "r1"


async def test_health_is_unauthenticated_and_minimal(tmp_path):
    seed(tmp_path, 1)
    async with client(make_app(tmp_path)) as c:
        r = await c.get("/v1/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "mode": "live", "journal": True}


async def test_missing_db_is_503_and_creates_nothing(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    app = make_app(state)
    async with client(app) as c:
        h = await c.get("/v1/health")
    assert h.status_code == 503
    assert h.json()["ok"] is False and h.json()["status"] == "degraded"
    st = await raw_stream(app, f"/v1/stream?token={TOKEN}")
    assert st["status"] == 503
    assert app.state.active["n"] == 0
    assert list(state.iterdir()) == []


async def test_unopenable_state_dir_is_503(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    app = make_app(blocker / "sub")
    st = await raw_stream(app, f"/v1/stream?token={TOKEN}")
    assert st["status"] == 503
    async with client(app) as c:
        assert (await c.get("/v1/health")).status_code == 503
    assert not (blocker / "sub").exists()


async def test_slot_released_when_response_never_starts(tmp_path):
    seed(tmp_path, 1)
    app = make_app(tmp_path)

    async def receive():
        return {"type": "http.disconnect"}

    async def send(msg):
        raise OSError("client gone")

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
        "scheme": "http", "path": "/v1/stream", "raw_path": b"/v1/stream",
        "query_string": f"token={TOKEN}".encode(), "headers": [(b"host", b"t")],
        "server": ("t", 80), "client": ("c", 1),
    }  # fmt: skip
    with pytest.raises(BaseException):  # noqa: B017
        await asyncio.wait_for(app(scope, receive, send), 5)
    assert app.state.active["n"] == 0


async def test_since_beyond_max_seq_keeps_streaming(tmp_path):
    j = seed(tmp_path, 2)
    app = make_app(tmp_path, heartbeat_s=0.05)
    st = await raw_stream(app, f"/v1/stream?token={TOKEN}&since=99", stop=lambda t: ": hb" in t)
    assert st["status"] == 200 and ids(st["text"]) == []
    del j


async def test_api_reads_the_journal_the_cli_writes(tmp_path):
    j = Journal(tmp_path / "events.sqlite")
    j.append("cli-run", "run.status", {"status": "running"})
    app = make_app(tmp_path)
    async with client(app) as c:
        runs = (await c.get(f"/v1/runs?token={TOKEN}")).json()["runs"]
    assert [r["run"] for r in runs] == ["cli-run"]
