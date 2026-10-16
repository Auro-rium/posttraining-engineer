from __future__ import annotations

import json
import os
import stat

import pytest

from posttrain_agent.observability.events import CURRENT_RUN, EVENT_TYPES, Journal, sanitize


def test_event_types_are_the_sixteen_in_the_spec():
    assert len(EVENT_TYPES) == 16  # the 15 of the spec + a2a.message
    assert CURRENT_RUN.get() == ""


def test_seq_is_monotonic_and_read_resumes_after_since(tmp_path):
    j = Journal(tmp_path / "events.sqlite")
    seqs = [j.append("r1", "checkpoint", {"checkpoint_id": f"c{i}", "step": i}) for i in range(5)]
    assert seqs == sorted(seqs) and len(set(seqs)) == 5
    assert j.last_seq() == seqs[-1]
    rest = j.read(since=seqs[1])
    assert [e.seq for e in rest] == seqs[2:]
    assert rest[0].run == "r1" and rest[0].type == "checkpoint"
    assert rest[0].data == {"checkpoint_id": "c2", "step": 2}
    j.append("r2", "checkpoint", {"checkpoint_id": "x", "step": 9})
    assert [e.run for e in j.read(run="r2")] == ["r2"]
    assert len(j.read(limit=3)) == 3


def test_unknown_type_is_rejected(tmp_path):
    j = Journal(tmp_path / "events.sqlite")
    assert sanitize("nope", {"a": 1}) is None
    assert j.append("r", "nope", {"a": 1}) is None
    assert j.read() == []


def test_non_allowlisted_fields_dropped_and_long_strings_truncated(tmp_path):
    j = Journal(tmp_path / "events.sqlite")
    j.append("r", "job.log", {"level": "info", "message": "x" * 1000, "prompt": "p"})
    (ev,) = j.read()
    assert set(ev.data) == {"level", "message"}
    assert len(ev.data["message"]) == 300


def test_sanitize_depth_limit():
    out = sanitize(
        "approval.pending",
        {"hyperparameters": {"lr": 1, "deep": {"x": {"y": 1}}}, "estimate_usd": 1.5},
    )
    assert out["estimate_usd"] == 1.5
    assert out["hyperparameters"]["lr"] == 1
    assert out["hyperparameters"] == {"lr": 1, "deep": {}}


def test_bad_keys_dropped_and_collections_capped(tmp_path):
    path = tmp_path / "events.sqlite"
    j = Journal(path)
    hp = {"SECRET text with spaces": 1, "k" * 49: 2, "ok-key.1_": 3}
    j.append("r", "approval.pending", {"hyperparameters": hp})
    big = {f"k{i}": i for i in range(100)}
    j.append("r", "gate.verdict", {"reasons": list(range(100)), "decision": "x"})
    j.append("r", "approval.pending", {"hyperparameters": big})
    a, b, c = j.read()
    assert a.data["hyperparameters"] == {"ok-key.1_": 3}
    assert len(b.data["reasons"]) == 32
    assert len(c.data["hyperparameters"]) == 32
    blob = b"".join(p.read_bytes() for p in tmp_path.iterdir())
    assert b"SECRET" not in blob


def test_non_finite_floats_become_null(tmp_path):
    j = Journal(tmp_path / "events.sqlite")
    j.append("r", "score", {"value": float("nan"), "n": float("inf"), "split": "a"})
    (ev,) = j.read()
    assert ev.data == {"value": None, "n": None, "split": "a"}
    import sqlite3

    def boom(c):
        raise ValueError(c)

    raw = sqlite3.connect(tmp_path / "events.sqlite").execute("SELECT data FROM events").fetchone()
    assert json.loads(raw[0], parse_constant=boom) == ev.data


def test_sentinel_strings_in_every_field_never_reach_the_journal(tmp_path):
    path = tmp_path / "events.sqlite"
    j = Journal(path)
    j.append("r", "tool.call", {"name": "t", "ms": 1, "ok": True, "args": {"q": "SECRET"},
                                "content": "SECRET", "messages": [{"content": "SECRET"}]})
    j.append("r", "model.call", {"name": "p:m", "prompt": "SECRET", "completion": "SECRET",
                                 "nested": {"a": {"b": {"c": "SECRET"}}}})
    j.append("r", "approval.pending", {"action": "launch_training",
                                       "hyperparameters": {"a": {"b": {"c": "SECRET"}}},
                                       "api_key": "SECRET"})
    assert j.read()
    blob = b"".join(p.read_bytes() for p in tmp_path.iterdir())
    assert b"SECRET" not in blob


def test_prune_keeps_latest_100000_rows(tmp_path):
    j = Journal(tmp_path / "events.sqlite", max_rows=10, prune_every=5)
    for i in range(23):
        j.append("r", "checkpoint", {"step": i})
    events = j.read(limit=1000)
    # prunes at writes 5,10,15,20: after write 20 rows 11..20 remain, then 21..23 added
    assert [e.data["step"] for e in events] == list(range(10, 23))
    assert events[-1].data["step"] == 22
    assert Journal(tmp_path / "x.sqlite").max_rows == 100_000


def test_runs_summary(tmp_path):
    j = Journal(tmp_path / "events.sqlite")
    j.append("r1", "run.started", {"goal_len": 3, "backend": "stub"})
    j.append("r1", "run.status", {"status": "RUNNING"})
    j.append("r1", "run.status", {"status": "COMPLETED"})
    (r,) = j.runs()
    assert r["run"] == "r1" and r["status"] == "COMPLETED"
    assert r["last_seq"] == j.last_seq() and r["started"] > 0


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_append_never_raises_when_db_is_unwritable(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        j = Journal(ro / "events.sqlite")
        assert j.append("r", "checkpoint", {"step": 1}) is None
        assert j.read() == []
        assert j.last_seq() == 0
    finally:
        ro.chmod(stat.S_IRWXU)
