"""Retry with exponential backoff and jitter for transient network failures.

Only transient errors are retried (transport/DNS/timeouts, HTTP 5xx and 429, the sandbox SDK's
``ApiTimeoutError``). Deterministic failures (bad input, a failing job script) are raised at once.
Delays are read at call time so tests can set ``BASE_DELAY = 0``.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx

T = TypeVar("T")
TRIES = 4
BASE_DELAY = 1.0
MAX_DELAY = 30.0
_NAMES = {"ApiTimeoutError", "ApiConnectionError", "ApiTransportError", "ConnectError",
          "TimeoutError", "ReadTimeout", "ConnectTimeout", "TooManyRequestsError"}


class TransientError(Exception):
    """A retryable failure that is not an exception from the transport (HTTP 5xx / 429)."""


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, TransientError | httpx.TransportError | TimeoutError | OSError):
        return True  # OSError covers socket.gaierror (DNS) and connection resets
    return any(c.__name__ in _NAMES for c in type(exc).__mro__)


def delay_for(attempt: int, base: float | None = None, cap: float | None = None) -> float:
    base = BASE_DELAY if base is None else base
    cap = MAX_DELAY if cap is None else cap
    return min(cap, base * 2 ** attempt) * (0.5 + random.random() / 2)


def retry_sync(fn: Callable[[], T], *, tries: int | None = None,
               sleep: Callable[[float], Any] = time.sleep,
               retry_on: Callable[[BaseException], bool] = is_transient) -> T:
    n = TRIES if tries is None else max(1, tries)
    for attempt in range(n):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified by retry_on
            if attempt == n - 1 or not retry_on(exc):
                raise
            sleep(delay_for(attempt))
    raise AssertionError("unreachable")


async def retry_async(fn: Callable[[], Awaitable[T]], *, tries: int | None = None,
                      retry_on: Callable[[BaseException], bool] = is_transient) -> T:
    n = TRIES if tries is None else max(1, tries)
    for attempt in range(n):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - classified by retry_on
            if attempt == n - 1 or not retry_on(exc):
                raise
            await asyncio.sleep(delay_for(attempt))
    raise AssertionError("unreachable")
