"""AWS transport boundaries for continuous trace events.

These adapters do not own validation, deduplication, or cycle policy.  They
only translate AWS envelopes into mappings accepted by ``TraceIngestor`` and
publish already-validated events.  Supplying a boto3 client explicitly keeps
local tests deterministic and avoids requiring AWS credentials at import time.
"""

import json
from collections.abc import Mapping
from typing import Any, Protocol

from .events import TraceEvent


class EventBridgeClient(Protocol):
    def put_events(self, *, Entries: list[dict[str, Any]]) -> Mapping[str, Any]: ...


class SQSClient(Protocol):
    def delete_message(self, *, QueueUrl: str, ReceiptHandle: str) -> Mapping[str, Any]: ...


def parse_eventbridge_event(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Extract a trace mapping from an EventBridge event envelope.

    EventBridge's stable ``id`` and ``time`` are fallbacks for producers that
    did not include ``event_id`` and ``occurred_at`` in ``detail``.  Detail
    values win, which preserves the producer's idempotency identity.
    """

    detail = envelope.get("detail", envelope)
    if not isinstance(detail, Mapping):
        raise ValueError("EventBridge detail must be an object")
    result = dict(detail)
    result.setdefault("event_id", envelope.get("id"))
    result.setdefault("occurred_at", envelope.get("time"))
    result.setdefault("source", envelope.get("source", "eventbridge"))
    result.setdefault("event_type", envelope.get("detail-type", "trace"))
    return result


def parse_sqs_records(envelope: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Decode SQS ``Records`` bodies into trace mappings.

    SQS is at-least-once; ``messageId`` is used only as a last-resort event id
    and the trace producer's id remains authoritative when present.
    """

    records = envelope.get("Records")
    if not isinstance(records, list):
        raise ValueError("SQS envelope must contain a Records list")
    parsed: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("SQS record must be an object")
        body = record.get("body", record)
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError as exc:
                raise ValueError("SQS record body must contain JSON") from exc
        if not isinstance(body, Mapping):
            raise ValueError("SQS record body must be an object")
        # An SQS body commonly contains an EventBridge envelope.  Reuse the
        # EventBridge translation so its event id/time/source remain intact.
        item = (
            parse_eventbridge_event(body)
            if isinstance(body.get("detail"), Mapping)
            else dict(body)
        )
        if item is None:
            raise ValueError("SQS record detail must be an object")
        item.setdefault("event_id", record.get("messageId"))
        parsed.append(item)
    return tuple(parsed)


class EventBridgePublisher:
    """Thin publisher boundary; the caller owns retries and cycle state."""

    def __init__(self, client: EventBridgeClient, *, event_bus_name: str, source: str) -> None:
        self.client = client
        self.event_bus_name = event_bus_name
        self.source = source

    def publish(self, event: TraceEvent, *, detail_type: str = "agent.trace") -> str:
        response = self.client.put_events(
            Entries=[
                {
                    "EventBusName": self.event_bus_name,
                    "Source": self.source,
                    "DetailType": detail_type,
                    "Detail": event.model_dump_json(by_alias=True),
                }
            ]
        )
        failed = int(response.get("FailedEntryCount", 0))
        if failed:
            raise RuntimeError(f"EventBridge rejected {failed} event(s)")
        entries = response.get("Entries", [])
        entry_id = (
            entries[0].get("EventId")
            if entries and isinstance(entries[0], Mapping)
            else None
        )
        return str(entry_id or event.event_id)


class SQSAckAdapter:
    """Explicit ack boundary for a successfully ingested SQS message."""

    def __init__(self, client: SQSClient, *, queue_url: str) -> None:
        self.client = client
        self.queue_url = queue_url

    def acknowledge(self, receipt_handle: str) -> None:
        self.client.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)


# Stable descriptive aliases for callers that refer to transports as
# adapters.  The concrete responsibilities remain intentionally narrow.
EventBridgeAdapter = EventBridgePublisher
SQSAdapter = SQSAckAdapter
