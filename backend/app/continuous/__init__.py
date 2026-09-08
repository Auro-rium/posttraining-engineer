"""Continuous trace ingestion primitives.

The package is deliberately independent from the FastAPI application and the
agent/provider implementations.  It can therefore be exercised locally with
the deterministic in-memory components, while the AWS adapters remain thin
boundaries around EventBridge and SQS clients.
"""

from .adapters import (
    EventBridgeAdapter,
    EventBridgePublisher,
    SQSAckAdapter,
    SQSAdapter,
    parse_eventbridge_event,
    parse_sqs_records,
)
from .deduplication import InMemoryDeduplicator
from .events import TraceEvent
from .ingestion import IngestionResult, InMemoryTraceStore, TraceIngestor
from .trigger import (
    CycleRequest,
    CycleTrigger,
    CycleTriggerConfig,
    CycleTriggerResult,
)

__all__ = [
    "CycleRequest",
    "CycleTrigger",
    "CycleTriggerConfig",
    "CycleTriggerResult",
    "EventBridgeAdapter",
    "EventBridgePublisher",
    "InMemoryDeduplicator",
    "InMemoryTraceStore",
    "IngestionResult",
    "SQSAckAdapter",
    "SQSAdapter",
    "TraceEvent",
    "TraceIngestor",
    "parse_eventbridge_event",
    "parse_sqs_records",
]
