from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from posttrain_agent.observability.events import Event
from posttrain_agent.observability.fold import empty_state, fold, snapshot

GOLDEN = json.loads((Path(__file__).parent.parent / "contract" / "golden.json").read_text())


@pytest.mark.parametrize("case", GOLDEN, ids=[c["name"] for c in GOLDEN])
def test_golden_fixtures(case: dict) -> None:
    state = empty_state()
    for ev in case["events"]:
        state = fold(state, ev)
    for key, want in case["expected"].items():
        assert state[key] == want, key
    for key, n in case.get("expected_len", {}).items():
        assert len(state[key]) == n, key
    for key, first in case.get("expected_first", {}).items():
        assert state[key][0] == first, key


def test_unknown_and_malformed_events_do_not_raise() -> None:
    state = empty_state()
    bad = [
        None,
        42,
        "x",
        [],
        {},
        {"seq": "1", "ts": 1, "run": "r", "type": "run.status", "data": {}},
        {"seq": 1, "ts": 1, "run": "r", "type": 7, "data": {}},
        {"seq": 1, "ts": 1, "run": "r", "type": "run.status", "data": "oops"},
        {"seq": 1, "ts": 1, "run": "r", "type": "nope", "data": {}},
        {"seq": 2, "ts": 1, "run": "r", "type": "agent.start", "data": {"role": 5, "slot": "x"}},
        {"seq": 3, "ts": 1, "run": "r", "type": "score", "data": {"value": object()}},
        Event(4, 1.0, "r", "run.status", None),  # type: ignore[arg-type]
    ]
    for ev in bad:
        state = fold(state, ev)  # type: ignore[arg-type]
    assert state == empty_state()


def test_snapshot_equals_incremental_fold() -> None:
    events = [
        Event(1, 1.0, "r", "run.started", {"backend": "stub", "goal_len": 10}),
        Event(2, 2.0, "r", "agent.start", {"role": "trainer", "slot": 1}),
        Event(3, 3.0, "r", "score", {"split": "val", "checkpoint": "c1", "value": 0.5}),
        Event(4, 4.0, "r", "agent.end", {"role": "trainer", "slot": 1, "ok": True}),
        Event(5, 5.0, "r", "run.status", {"status": "COMPLETED", "attempt": 1}),
    ]
    state = empty_state()
    for ev in events:
        state = fold(state, ev)
    assert snapshot(events) == state
    assert state["last_seq"] == 5
    assert state["status"] == "COMPLETED"


def test_fold_does_not_mutate_input() -> None:
    state = fold(
        empty_state(),
        {"seq": 1, "ts": 1, "run": "r", "type": "agent.start", "data": {"role": "a", "slot": 1}},
    )
    before = copy.deepcopy(state)
    new = fold(
        state,
        {"seq": 2, "ts": 2, "run": "r", "type": "agent.end", "data": {"role": "a", "slot": 1}},
    )
    assert state == before
    assert new is not state
    assert new != state
