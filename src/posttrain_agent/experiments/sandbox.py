"""Run the game-environment judge inside a Nebius Sandbox (contree-sdk).

The model's tool calls are produced by Token Factory inference on the host; they reach the sandbox
only as data. The sandbox gets the judge source and the calls, executes them against the game
rules in an isolated VM and returns verdicts. No API key ever enters the sandbox.

Sandboxes are a Nebius Beta product: the project must have access (contact contree@nebius.com
with the project id) and ``NEBIUS_PROJECT_ID`` must be set. Requests go out in small batches of
one shell command each; the judge source and the batch are passed base64-encoded because the
published docs do not show a file-upload call.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import threading
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from ..config import Settings
from . import game_env
from .retry import is_transient, retry_async

BATCH = 40  # calls per sandbox operation
PARALLEL = 4  # operations in flight (Beta allows 50 per project)
TIMEOUT_NOTE = "sandbox operation failed"
TRANSPORT_TIMEOUT_S = 600.0  # SDK default 10 s timed out uploading a LoRA adapter (live)


class SandboxUnavailable(RuntimeError):
    """The sandbox cannot be used (SDK missing, project not configured, or a call failed).

    ``transient`` is True when retries were exhausted on a network/timeout failure."""

    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


def sandbox_configured(settings: Settings) -> bool:
    return bool(settings.nebius_api_key and settings.nebius_project_id
                and importlib.util.find_spec("contree_sdk"))


def contree_client(settings: Settings, transport_timeout: float | None = None,
                   operation_timeout: float | None = None) -> Any:
    """A host-side ConTree client. The key reaches the SDK only through this process' environment;
    it is never put into a sandbox command, its stdin, its files or its environment.

    ``transport_timeout`` overrides the SDK's 10 s per-request timeout, which cuts multi-MB file
    uploads (seen live on a LoRA adapter: ``ApiTimeoutError`` on ``POST /v1/files``)."""
    try:
        from contree_sdk import Contree
        from contree_sdk.config import ContreeConfig
    except ImportError as exc:
        raise SandboxUnavailable("contree-sdk is not installed (pip install '.[sandbox]')") from exc
    if not (settings.nebius_api_key and settings.nebius_project_id):
        raise SandboxUnavailable("NEBIUS_API_KEY and NEBIUS_PROJECT_ID are required")
    os.environ.setdefault("NEBIUS_API_KEY", settings.nebius_api_key)  # the SDK reads these
    os.environ.setdefault("NEBIUS_PROJECT_ID", settings.nebius_project_id)
    overrides = {k: v for k, v in (("transport_timeout", transport_timeout),
                                   ("operation_timeout", operation_timeout)) if v is not None}
    return Contree(ContreeConfig(**overrides)) if overrides else Contree()


def _run_sync(make: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    """Run a coroutine to completion from sync code, even when a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make())
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = asyncio.run(make())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
            box["error"] = exc

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


class SandboxJudge:
    def __init__(self, settings: Settings, client_factory: Callable[[], Any] | None = None) -> None:
        self.s = settings
        self._factory = client_factory
        self._src = base64.b64encode(Path(game_env.__file__).read_bytes()).decode()

    def _client(self) -> Any:
        return self._factory() if self._factory else contree_client(
            self.s, transport_timeout=TRANSPORT_TIMEOUT_S)

    def judge(self, items: list[dict]) -> list[dict]:
        """Verdicts ``[{id, ok, reason}]`` for ``[{id, state, name, arguments}]``, in order."""
        if not items:
            return []
        return _run_sync(lambda: self._judge(items))

    async def _judge(self, items: list[dict]) -> list[dict]:
        client = self._client()
        try:
            image = await retry_async(lambda: client.images.use(self.s.sandbox_image, strict=True))
        except Exception as exc:  # noqa: BLE001 - any SDK/network failure
            raise SandboxUnavailable(f"{TIMEOUT_NOTE}: images.use: {type(exc).__name__}",
                                     transient=is_transient(exc)) from exc
        gate = asyncio.Semaphore(PARALLEL)

        async def one(chunk: list[dict]) -> list[dict]:
            async with gate:
                payload = base64.b64encode(json.dumps({"items": chunk}).encode()).decode()
                shell = (f"printf %s {self._src} | base64 -d > /tmp/game_env.py && "
                         f"printf %s {payload} | base64 -d | python3 /tmp/game_env.py")
                try:  # transient transport failures are retried per chunk; verdicts stay pure
                    res = await retry_async(
                        lambda: image.run(shell=shell, disposable=True, timeout=120))
                except Exception as exc:  # noqa: BLE001 - any SDK/network failure
                    raise SandboxUnavailable(f"{TIMEOUT_NOTE}: {type(exc).__name__}",
                                             transient=is_transient(exc)) from exc
                if res.exit_code != 0:
                    raise SandboxUnavailable(f"{TIMEOUT_NOTE}: exit {res.exit_code}: "
                                             f"{(res.stderr or '')[:200]}")
                try:
                    results = json.loads(res.stdout)["results"]
                except (ValueError, KeyError, TypeError) as exc:
                    raise SandboxUnavailable("sandbox returned unreadable output") from exc
                if [r.get("id") for r in results] != [c.get("id") for c in chunk]:
                    raise SandboxUnavailable("sandbox verdicts do not match the submitted calls")
                return results

        chunks = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
        return [r for part in await asyncio.gather(*(one(c) for c in chunks)) for r in part]
