"""Error fallback: retry transient Backboard errors, then fall through to other models.

Middleware order matters. ``ModelFallbackMiddleware`` is outermost and
``ModelRetryMiddleware`` sits inside it, so the primary model is retried with
backoff, and *each* fallback model gets its own retries before the next one.
"""

from __future__ import annotations

from backboard import (
    BackboardAPIError,
    BackboardNotFoundError,
    BackboardRateLimitError,
    BackboardServerError,
    BackboardValidationError,
)
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)

from .backboard_model import CassetteMiss, ChatBackboard
from .config import Settings


def is_transient(exc: Exception) -> bool:
    """True for errors worth retrying: rate limits, 5xx, timeouts/network, empty replies.

    The SDK collapses timeouts and connection errors into a ``BackboardAPIError``
    with no status code, so a missing status counts as transient.
    """
    if isinstance(exc, CassetteMiss | BackboardValidationError | BackboardNotFoundError):
        return False
    if isinstance(exc, BackboardRateLimitError | BackboardServerError):
        return True
    if isinstance(exc, BackboardAPIError):
        code = exc.status_code
        return code is None or code >= 500
    return False


def build_model(settings: Settings, *, provider: str | None = None, model: str | None = None,
                **extra) -> ChatBackboard:
    return ChatBackboard(
        api_key=settings.backboard_api_key,
        base_url=settings.backboard_base_url,
        assistant_id=settings.assistant_id,
        llm_provider=provider or settings.llm_provider,
        model_name=model or settings.model_name,
        memory=settings.memory_mode,
        timeout=settings.request_timeout,
        max_input_tokens=settings.max_input_tokens,
        **extra,
    )


def build_middleware(settings: Settings, **model_extra) -> list:
    """Fallback (outer) -> model retry -> tool retry."""
    stack: list = []
    fallbacks = [
        build_model(settings, provider=p, model=m, **model_extra)
        for p, m in settings.fallback_models
    ]
    if fallbacks:
        stack.append(ModelFallbackMiddleware(*fallbacks))
    stack.append(
        ModelRetryMiddleware(
            max_retries=settings.model_retries,
            retry_on=is_transient,
            on_failure="error",
            initial_delay=1.0,
            max_delay=20.0,
        )
    )
    stack.append(ToolRetryMiddleware(max_retries=1, on_failure="continue"))
    return stack
