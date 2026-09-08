"""Validated trace-event contract used by continuous ingestion."""

from datetime import UTC, datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class TraceEvent(BaseModel):
    """A single immutable observation emitted by an agent run.

    ``event_id`` is the idempotency key.  Producers should generate it once
    and preserve it when retrying delivery through EventBridge or SQS.  The
    aliases accept both the Python-friendly snake_case form and the existing
    API's camelCase form.
    """

    model_config = ConfigDict(
        alias_generator=None,
        populate_by_name=True,
        extra="allow",
        frozen=True,
    )

    event_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        validation_alias=AliasChoices("event_id", "eventId", "id"),
        serialization_alias="eventId",
    )
    trace_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        validation_alias=AliasChoices("trace_id", "traceId"),
        serialization_alias="traceId",
    )
    run_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        validation_alias=AliasChoices("run_id", "runId"),
        serialization_alias="runId",
    )
    event_type: str = Field(
        ...,
        min_length=1,
        max_length=128,
        validation_alias=AliasChoices("event_type", "eventType", "type"),
        serialization_alias="eventType",
    )
    occurred_at: datetime = Field(
        ...,
        validation_alias=AliasChoices("occurred_at", "occurredAt", "timestamp", "time"),
        serialization_alias="occurredAt",
    )
    source: str = Field(default="agent", min_length=1, max_length=256)
    status: str | None = Field(default=None, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id", "trace_id", "run_id", "event_type", "source", "status")
    @classmethod
    def reject_blank_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value.astimezone(UTC)

    @field_validator("payload", "metadata")
    @classmethod
    def require_object(cls, value: dict[str, Any]) -> dict[str, Any]:
        # Pydantic already checks the type; this explicit copy prevents a
        # caller from mutating nested top-level mappings after validation.
        return dict(value)
