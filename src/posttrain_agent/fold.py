"""Pure snapshot reducer shared (by golden fixtures) with the TypeScript dashboard reducer.

State is plain JSON. ``fold`` never mutates its input, never raises, and ignores unknown or
malformed events entirely (state, ``log`` and ``last_seq`` are left untouched). An event is
malformed when ``seq`` is not an int, ``type`` is not a string, ``data`` is not a dict, ``data``
holds a non-JSON value, or a required field for its type is missing or mistyped. An event whose
``seq`` is not greater than ``last_seq`` is a duplicate or stale replay and is ignored too.

Per-type effects:
  run.started          backend, experiment={goal_len, experiment_budget} (present keys only)
  run.status           status (str), attempt (int), when present
  agent.start          agents[role]={active: true, slot, ok: null}
  agent.end            agents[role]={active: false, slot, ok: data.ok if bool else null}
  score                append {split, checkpoint, value, n, evidence} (present keys only)
  gate.verdict         gate replaced by the allow-listed fields present
  budget.update        budget merged key by key
  job.update           job merged key by key
  job.log              append {level, message} to job_log (last 50)
  approval.pending     approval = data;  approval.resolved: approval = null
  checkpoint           append {checkpoint_id, step}
  telemetry.snapshot   telemetry merged key by key
  model.call, tool.call  log only
``active_slots`` is the sorted list of slots of active agents.

Log summary format (the TypeScript reducer must match byte for byte). ``log`` keeps the last 200
entries ``{seq, ts, type, summary}``. summary is the ``key=value`` pairs for the fields listed
below, in this order, joined by one space, skipping fields that are absent or not a str, int or
bool. Values: str as is, int as decimal, bool as ``true``/``false``. Floats and lists are never
in a summary. Fields per type:
  run.started: backend | run.status: status attempt | agent.start: role slot
  agent.end: role slot ok | model.call: name ok fallback | tool.call: name ok
  score: split checkpoint evidence | gate.verdict: decision | budget.update: price_known
  job.update: job_id status | job.log: level message | approval.pending: action
  approval.resolved: decision | checkpoint: checkpoint_id step | telemetry.snapshot: (empty)
"""

from __future__ import annotations

import copy
import math
from typing import Any

from posttrain_agent.events import EVENT_TYPES, Event

LOG_MAX = 200
JOB_LOG_MAX = 50

_SUMMARY_FIELDS: dict[str, tuple[str, ...]] = {
    "run.started": ("backend",),
    "run.status": ("status", "attempt"),
    "agent.start": ("role", "slot"),
    "agent.end": ("role", "slot", "ok"),
    "model.call": ("name", "ok", "fallback"),
    "tool.call": ("name", "ok"),
    "score": ("split", "checkpoint", "evidence"),
    "gate.verdict": ("decision",),
    "budget.update": ("price_known",),
    "job.update": ("job_id", "status"),
    "job.log": ("level", "message"),
    "approval.pending": ("action",),
    "approval.resolved": ("decision",),
    "checkpoint": ("checkpoint_id", "step"),
    "telemetry.snapshot": (),
}
_SCORE_KEYS = ("split", "checkpoint", "value", "n", "evidence")
_GATE_KEYS = ("decision", "reasons", "champion_score", "candidate_score", "relative_improvement")


def empty_state() -> dict:
    return {
        "status": None,
        "attempt": 0,
        "backend": None,
        "experiment": None,
        "agents": {},
        "active_slots": [],
        "log": [],
        "scores": [],
        "gate": None,
        "budget": {},
        "job": {},
        "job_log": [],
        "approval": None,
        "telemetry": {},
        "checkpoints": [],
        "last_seq": 0,
    }


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_json(v: Any, depth: int = 0) -> bool:
    if depth > 8:
        return False
    if v is None or isinstance(v, (bool, int, str)):
        return True
    if isinstance(v, float):
        return math.isfinite(v)
    if isinstance(v, list):
        return all(_is_json(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_json(x, depth + 1) for k, x in v.items())
    return False


def _parse(ev: Any) -> tuple[int, Any, str, dict] | None:
    if isinstance(ev, Event):
        seq, ts, typ, data = ev.seq, ev.ts, ev.type, ev.data
    elif isinstance(ev, dict):
        seq, ts, typ, data = ev.get("seq"), ev.get("ts"), ev.get("type"), ev.get("data")
    else:
        return None
    if not _is_int(seq) or not isinstance(typ, str) or typ not in EVENT_TYPES:
        return None
    if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not _is_json(ts):
        return None
    if not isinstance(data, dict) or not _is_json(data):
        return None
    return seq, ts, typ, data


def _summary(typ: str, data: dict) -> str:
    parts = []
    for key in _SUMMARY_FIELDS[typ]:
        v = data.get(key)
        if isinstance(v, bool):
            parts.append(f"{key}={'true' if v else 'false'}")
        elif isinstance(v, (int, str)):
            parts.append(f"{key}={v}")
    return " ".join(parts)


def _pick(data: dict, keys: tuple[str, ...]) -> dict:
    return {k: data[k] for k in keys if k in data}


def _apply(s: dict, typ: str, data: dict) -> bool:
    """Apply one event to the (already copied) state. False means malformed: discard."""
    if typ == "run.started":
        if isinstance(data.get("backend"), str):
            s["backend"] = data["backend"]
        s["experiment"] = _pick(data, ("goal_len", "experiment_budget"))
    elif typ == "run.status":
        if isinstance(data.get("status"), str):
            s["status"] = data["status"]
        if _is_int(data.get("attempt")):
            s["attempt"] = data["attempt"]
    elif typ in ("agent.start", "agent.end"):
        role, slot = data.get("role"), data.get("slot")
        if not isinstance(role, str) or not _is_int(slot):
            return False
        if typ == "agent.start":
            s["agents"][role] = {"active": True, "slot": slot, "ok": None}
        else:
            ok = data.get("ok")
            s["agents"][role] = {
                "active": False,
                "slot": slot,
                "ok": ok if isinstance(ok, bool) else None,
            }
        s["active_slots"] = sorted(a["slot"] for a in s["agents"].values() if a["active"])
    elif typ == "score":
        s["scores"].append(_pick(data, _SCORE_KEYS))
    elif typ == "gate.verdict":
        s["gate"] = _pick(data, _GATE_KEYS)
    elif typ == "budget.update":
        s["budget"].update(data)
    elif typ == "job.update":
        s["job"].update(data)
    elif typ == "job.log":
        s["job_log"].append(_pick(data, ("level", "message")))
        del s["job_log"][:-JOB_LOG_MAX]
    elif typ == "approval.pending":
        s["approval"] = dict(data)
    elif typ == "approval.resolved":
        s["approval"] = None
    elif typ == "checkpoint":
        s["checkpoints"].append(_pick(data, ("checkpoint_id", "step")))
    elif typ == "telemetry.snapshot":
        s["telemetry"].update(data)
    return True


def fold(state: dict, ev: Event | dict) -> dict:
    """Return a new state with ``ev`` applied; the input state is never mutated."""
    try:
        parsed = _parse(ev)
        if parsed is None:
            return copy.deepcopy(state)
        seq, ts, typ, data = parsed
        if seq <= state.get("last_seq", 0):
            return copy.deepcopy(state)
        s = copy.deepcopy(state)
        if not _apply(s, typ, data):
            return copy.deepcopy(state)
        s["log"].append({"seq": seq, "ts": ts, "type": typ, "summary": _summary(typ, data)})
        del s["log"][:-LOG_MAX]
        s["last_seq"] = seq
        return s
    except Exception:  # noqa: BLE001 - the reducer must never raise
        return copy.deepcopy(state)


def snapshot(events: list[Event]) -> dict:
    state = empty_state()
    for ev in events:
        state = fold(state, ev)
    return state
