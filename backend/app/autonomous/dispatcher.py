"""Lease-based dispatcher for recoverable autonomous runs.

The dispatcher is intentionally small: DynamoDB/in-memory repositories own
the conditional lease, while the supervisor owns all phase transitions.  A
second worker can observe the same run but cannot execute it while the lease
is valid.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from typing import Any, Protocol, cast

from .models import AutonomousRunState
from .repository import AutonomousRunRepository, LeaseConflictError


class OptimizationRunner(Protocol):
    def run_optimization(
        self, run_id: str
    ) -> AutonomousRunState | Awaitable[AutonomousRunState]: ...


async def _await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class AutonomousRunDispatcher:
    """Claim and execute each currently recoverable run at most once."""

    def __init__(
        self,
        *,
        repository: AutonomousRunRepository,
        supervisor: OptimizationRunner,
        owner: str,
        lease_ttl_seconds: int = 60,
        scan_limit: int = 100,
    ) -> None:
        if not owner or not owner.strip():
            raise ValueError("owner must not be blank")
        if lease_ttl_seconds <= 0 or scan_limit <= 0:
            raise ValueError("lease and scan limits must be positive")
        self.repository = repository
        self.supervisor = supervisor
        self.owner = owner
        self.lease_ttl_seconds = lease_ttl_seconds
        self.scan_limit = scan_limit
        if getattr(supervisor, "owner", None) is None:
            cast(Any, supervisor).owner = owner

    async def dispatch_once(self) -> list[str]:
        page = self.repository.scan_recoverable(limit=self.scan_limit)
        processed: list[str] = []
        for candidate in page:
            try:
                self.repository.claim_lease(
                    candidate.run_id, self.owner, ttl_seconds=self.lease_ttl_seconds
                )
            except LeaseConflictError:
                continue
            try:
                await self._run_with_lease(candidate.run_id)
                processed.append(candidate.run_id)
            finally:
                current = self.repository.get(candidate.run_id)
                if current is not None and current.lease_owner == self.owner:
                    self.repository.release_lease(candidate.run_id, self.owner)
        return processed

    async def _run_with_lease(self, run_id: str) -> None:
        """Run with a bounded heartbeat; cancel work if ownership is lost."""

        interval = max(0.05, self.lease_ttl_seconds / 3)
        lost = asyncio.Event()

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(interval)
                try:
                    self.repository.renew_lease(
                        run_id, self.owner, ttl_seconds=self.lease_ttl_seconds
                    )
                except Exception:
                    lost.set()
                    return

        work = asyncio.create_task(_await(self.supervisor.run_optimization(run_id)))
        beat = asyncio.create_task(heartbeat())
        lost_wait = asyncio.create_task(lost.wait())
        try:
            done, _ = await asyncio.wait({work, lost_wait}, return_when=asyncio.FIRST_COMPLETED)
            if lost_wait in done and lost.is_set():
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
                raise LeaseConflictError("dispatcher lease was lost")
            await work
        finally:
            beat.cancel()
            lost_wait.cancel()
            await asyncio.gather(beat, lost_wait, return_exceptions=True)

    async def run_once(self) -> list[str]:
        """Compatibility alias for queue workers."""

        return await self.dispatch_once()

    async def recover_incomplete_runs(self) -> list[str]:
        """Resume queued, crashed, or lease-expired runs."""

        return await self.dispatch_once()


__all__ = ["AutonomousRunDispatcher", "OptimizationRunner"]
