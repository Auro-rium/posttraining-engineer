import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.continuous import (
    CycleTrigger,
    CycleTriggerConfig,
    InMemoryDeduplicator,
    TraceEvent,
    TraceIngestor,
)
from app.continuous.adapters import parse_eventbridge_event, parse_sqs_records


def event(event_id: str, *, event_type: str = "step", status: str | None = None) -> dict:
    return {
        "event_id": event_id,
        "trace_id": "trace-1",
        "run_id": "run-1",
        "event_type": event_type,
        "occurred_at": datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        "status": status,
        "payload": {"step": event_id},
    }


def test_trace_event_validates_and_normalizes_aliases() -> None:
    trace_event = TraceEvent.model_validate(
        {
            "eventId": "evt-1",
            "traceId": "trace-1",
            "runId": "run-1",
            "eventType": "step",
            "occurredAt": "2026-01-01T12:00:00+05:30",
        }
    )
    assert trace_event.event_id == "evt-1"
    assert trace_event.occurred_at.isoformat() == "2026-01-01T06:30:00+00:00"

    with pytest.raises(ValidationError):
        TraceEvent.model_validate({"event_id": "evt-1"})


def test_duplicate_delivery_is_not_persisted_or_counted() -> None:
    ingestor = TraceIngestor(
        deduplicator=InMemoryDeduplicator(),
        trigger=CycleTrigger(CycleTriggerConfig(event_threshold=2)),
    )
    first = ingestor.ingest(event("evt-1"))
    duplicate = ingestor.ingest(event("evt-1"))
    second = ingestor.ingest(event("evt-2"))

    assert first.accepted is True
    assert duplicate.accepted is False and duplicate.duplicate is True
    assert second.cycle is not None
    assert second.cycle.event_ids == ("evt-1", "evt-2")
    assert len(ingestor.store) == 2


def test_failure_threshold_and_per_run_isolation() -> None:
    trigger = CycleTrigger(CycleTriggerConfig(event_threshold=2, failure_threshold=1))
    assert trigger.offer(TraceEvent.model_validate(event("a-1"))).triggered is False
    assert trigger.offer(TraceEvent.model_validate(event("a-2", status="failed"))).triggered is True

    other = event("b-1") | {"run_id": "run-2"}
    assert trigger.offer(TraceEvent.model_validate(other)).pending_count == 1


def test_aws_envelope_parsers() -> None:
    eventbridge = parse_eventbridge_event(
        {
            "id": "eb-1",
            "source": "agent.service",
            "time": "2026-01-01T12:00:00Z",
            "detail": {
                "traceId": "trace-1",
                "runId": "run-1",
                "eventType": "step",
            },
        }
    )
    assert TraceEvent.model_validate(eventbridge).event_id == "eb-1"

    records = parse_sqs_records(
        {
            "Records": [
                {
                    "messageId": "sqs-1",
                    "body": json.dumps(event("producer-event"), default=str),
                }
            ]
        }
    )
    assert records[0]["event_id"] == "producer-event"
