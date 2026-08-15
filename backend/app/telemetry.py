"""OpenTelemetry helpers that never attach prompts or task content to spans."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from enum import Enum
from typing import Any

SAFE_ATTRIBUTE_NAMES = frozenset(
    {
        "run_id",
        "experiment_id",
        "agent_name",
        "a2a_task_id",
        "artifact_type",
        "phase",
        "operation",
        "status",
        "error_code",
        "model_id",
        "job_id",
        "duration_ms",
        "token_count",
        "retrieval_count",
        "handoff_count",
        "task_count",
        "retry_count",
    }
)

_SENSITIVE_NAME_PARTS = (
    "prompt",
    "content",
    "observation",
    "trajectory",
    "task_body",
    "secret",
    "token_value",
    "password",
    "credential",
    "api_key",
)


def safe_attributes(attributes: Mapping[str, Any] | None) -> dict[str, str | bool | int | float]:
    """Return only allow-listed scalar metadata suitable for export."""

    cleaned: dict[str, str | bool | int | float] = {}
    for key, value in (attributes or {}).items():
        normalized_key = key.lower()
        if any(part in normalized_key for part in _SENSITIVE_NAME_PARTS):
            continue
        if key not in SAFE_ATTRIBUTE_NAMES or value is None:
            continue
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
    return cleaned


class NoOpSpan:
    """Minimal span used when OpenTelemetry is not installed."""

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def record_exception(self, exception: BaseException) -> None:
        return None

    def set_status(self, status: Any) -> None:
        return None


def _set_attributes(span: Any, attributes: Mapping[str, Any] | None) -> None:
    for key, value in safe_attributes(attributes).items():
        span.set_attribute(key, value)


@contextmanager
def telemetry_span(
    name: str, *, attributes: Mapping[str, Any] | None = None
) -> Iterator[Any]:
    """Create a metadata-only span, or a no-op span without the optional SDK."""

    try:
        from opentelemetry import trace
    except ImportError:
        yield NoOpSpan()
        return
    tracer = trace.get_tracer("autonomous-post-training-engineer")
    with tracer.start_as_current_span(name) as span:
        _set_attributes(span, attributes)
        yield span


@asynccontextmanager
async def async_telemetry_span(
    name: str, *, attributes: Mapping[str, Any] | None = None
) -> AsyncIterator[Any]:
    """Async wrapper around :func:`telemetry_span`."""

    with telemetry_span(name, attributes=attributes) as span:
        yield span


def current_trace_id() -> str | None:
    """Return the active trace ID without requiring telemetry at runtime."""

    try:
        from opentelemetry import trace
    except ImportError:
        return None
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return f"{context.trace_id:032x}"


def configure_cloud_trace(*, project_id: str, service_name: str) -> None:
    """Configure Google Cloud Trace export when optional telemetry packages exist."""

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:  # pragma: no cover - optional cloud dependencies
        raise RuntimeError(
            "Cloud Trace export requires OpenTelemetry SDK and the GCP trace exporter"
        ) from exc
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": service_name, "gcp.project_id": project_id}
        )
    )
    provider.add_span_processor(
        BatchSpanProcessor(CloudTraceSpanExporter(project_id=project_id))  # type: ignore[no-untyped-call]
    )
    trace.set_tracer_provider(provider)
