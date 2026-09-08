"""Deterministic idempotency support for at-least-once delivery."""

from threading import RLock

from .events import TraceEvent


class InMemoryDeduplicator:
    """Thread-safe event-id set with an atomic claim operation.

    This is intentionally bounded to the process lifetime.  Production
    wiring should replace it with a durable conditional-write store (for
    example DynamoDB), but must retain the same ``claim`` semantics.
    """

    def __init__(self) -> None:
        self._event_ids: set[str] = set()
        self._lock = RLock()

    def claim(self, event: TraceEvent) -> bool:
        """Atomically claim ``event``; return ``False`` when already seen."""

        with self._lock:
            if event.event_id in self._event_ids:
                return False
            self._event_ids.add(event.event_id)
            return True

    def contains(self, event_id: str) -> bool:
        with self._lock:
            return event_id in self._event_ids

    def __len__(self) -> int:
        with self._lock:
            return len(self._event_ids)

    def clear(self) -> None:
        with self._lock:
            self._event_ids.clear()
