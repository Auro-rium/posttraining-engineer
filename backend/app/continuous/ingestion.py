"""Continuous validation, deduplication, storage, and cycle triggering."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .adapters import parse_eventbridge_event, parse_sqs_records
from .deduplication import InMemoryDeduplicator
from .events import TraceEvent
from .trigger import CycleRequest, CycleTrigger, CycleTriggerResult


class InMemoryTraceStore:
    """Ordered in-memory event store used by local tests and simulations."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def append(self, event: TraceEvent) -> None:
        self._events.append(event)

    def all(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def for_run(self, run_id: str) -> tuple[TraceEvent, ...]:
        return tuple(event for event in self._events if event.run_id == run_id)

    def __len__(self) -> int:
        return len(self._events)


@dataclass(frozen=True)
class IngestionResult:
    accepted: bool
    duplicate: bool = False
    error: str | None = None
    event: TraceEvent | None = None
    trigger: CycleTriggerResult | None = None

    @property
    def cycle(self) -> CycleRequest | None:
        return self.trigger.cycle if self.trigger else None


class TraceIngestor:
    """Synchronous deterministic ingestion pipeline.

    The order is intentional: validate, claim idempotency, persist, then
    offer to the trigger.  A duplicate is never counted toward a cycle.
    """

    def __init__(
        self,
        *,
        deduplicator: InMemoryDeduplicator | None = None,
        store: InMemoryTraceStore | None = None,
        trigger: CycleTrigger | None = None,
    ) -> None:
        self.deduplicator = deduplicator or InMemoryDeduplicator()
        self.store = store or InMemoryTraceStore()
        self.trigger = trigger or CycleTrigger()

    def ingest(self, raw_event: TraceEvent | Mapping[str, Any]) -> IngestionResult:
        try:
            event = (
                raw_event
                if isinstance(raw_event, TraceEvent)
                else TraceEvent.model_validate(raw_event)
            )
        except ValidationError as exc:
            return IngestionResult(accepted=False, error=_validation_error(exc))

        if not self.deduplicator.claim(event):
            return IngestionResult(accepted=False, duplicate=True, event=event)

        self.store.append(event)
        trigger_result = self.trigger.offer(event)
        return IngestionResult(accepted=True, event=event, trigger=trigger_result)

    def ingest_many(
        self, events: Iterable[TraceEvent | Mapping[str, Any]]
    ) -> tuple[IngestionResult, ...]:
        return tuple(self.ingest(event) for event in events)

    def ingest_eventbridge(self, envelope: Mapping[str, Any]) -> IngestionResult:
        """Translate and ingest one EventBridge envelope."""

        return self.ingest(parse_eventbridge_event(envelope))

    def ingest_sqs(self, envelope: Mapping[str, Any]) -> tuple[IngestionResult, ...]:
        """Translate and ingest all records in one SQS Lambda envelope."""

        return self.ingest_many(parse_sqs_records(envelope))


def _validation_error(error: ValidationError) -> str:
    # Keep API/log output stable and avoid exposing full arbitrary payloads.
    first = error.errors()[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "event"
    return f"invalid {location}: {first.get('msg', 'validation failed')}"
