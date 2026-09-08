"""Threshold-based, deterministic optimization-cycle triggering."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field

from .events import TraceEvent


class CycleTriggerConfig(BaseModel):
    """Controls how many accepted events form one optimization cycle."""

    model_config = ConfigDict(frozen=True)

    event_threshold: int = Field(default=10, ge=1)
    # If set, a cycle also requires at least this many failed/error events.
    failure_threshold: int | None = Field(default=None, ge=1)
    window_seconds: int | None = Field(default=None, ge=1)


class CycleRequest(BaseModel):
    """A deterministic request to start one post-training optimization cycle."""

    model_config = ConfigDict(frozen=True)

    cycle_id: str
    run_id: str
    event_ids: tuple[str, ...]
    reason: str
    triggered_at: datetime


class CycleTriggerResult(BaseModel):
    """Outcome of offering one validated event to the cycle trigger."""

    model_config = ConfigDict(frozen=True)

    accepted: bool = True
    triggered: bool = False
    pending_count: int = 0
    cycle: CycleRequest | None = None


@dataclass
class _PendingRun:
    events: list[TraceEvent]


class CycleTrigger:
    """Accumulates events per run and emits a request at a configured threshold.

    Once a cycle is emitted, its events are removed.  This makes repeated
    delivery safe when paired with :class:`InMemoryDeduplicator`, and allows
    subsequent events to form the next cycle without retriggering forever.
    """

    def __init__(self, config: CycleTriggerConfig | None = None) -> None:
        self.config = config or CycleTriggerConfig()
        self._pending: dict[str, _PendingRun] = {}

    def offer(self, event: TraceEvent) -> CycleTriggerResult:
        pending = self._pending.setdefault(event.run_id, _PendingRun(events=[]))
        pending.events.append(event)
        self._expire_old(pending, max(item.occurred_at for item in pending.events))

        failures = sum(1 for item in pending.events if _is_failure(item))
        threshold_reached = len(pending.events) >= self.config.event_threshold
        failure_reached = (
            self.config.failure_threshold is None
            or failures >= self.config.failure_threshold
        )
        if not (threshold_reached and failure_reached):
            return CycleTriggerResult(pending_count=len(pending.events))

        events = tuple(pending.events)
        cycle = CycleRequest(
            cycle_id=_cycle_id(event.run_id, events),
            run_id=event.run_id,
            event_ids=tuple(item.event_id for item in events),
            reason=(
                f"accepted_event_threshold={len(events)}"
                f";failure_count={failures}"
            ),
            triggered_at=max(item.occurred_at for item in events),
        )
        pending.events.clear()
        return CycleTriggerResult(triggered=True, pending_count=0, cycle=cycle)

    def pending_count(self, run_id: str) -> int:
        return len(self._pending.get(run_id, _PendingRun(events=[])).events)

    def clear(self, run_id: str | None = None) -> None:
        if run_id is None:
            self._pending.clear()
        else:
            self._pending.pop(run_id, None)

    def _expire_old(self, pending: _PendingRun, latest: datetime) -> None:
        if self.config.window_seconds is None:
            return
        cutoff = latest - timedelta(seconds=self.config.window_seconds)
        pending.events[:] = [event for event in pending.events if event.occurred_at >= cutoff]


def _is_failure(event: TraceEvent) -> bool:
    values = {event.event_type.lower(), (event.status or "").lower()}
    return bool(values & {"failure", "failed", "error", "exception"})


def _cycle_id(run_id: str, events: tuple[TraceEvent, ...]) -> str:
    material = "|".join((run_id, *(event.event_id for event in events)))
    return f"cycle-{sha256(material.encode('utf-8')).hexdigest()[:32]}"
