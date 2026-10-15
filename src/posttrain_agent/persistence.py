"""Durable state: LangGraph checkpoints + a cross-thread store, both on SQLite."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

MEMORY_NS = ("posttrain", "memories")
MEMORY_PREFIX = "/memories/"
SEED_MEMORY = f"{MEMORY_PREFIX}AGENTS.md"

SEED_TEXT = """# Standing knowledge

- Held-out tasks are sealed: never use them for hypotheses, data or training.
- Only deterministic code decides promotion; report evidence labels honestly.
- Record durable lessons (what regressed, what helped) in /memories/lessons.md.
"""


@dataclass
class Runtime:
    checkpointer: AsyncSqliteSaver
    store: AsyncSqliteStore
    backend: CompositeBackend


def build_backend(store: AsyncSqliteStore) -> CompositeBackend:
    """Scratch files live in graph state; ``/memories/`` persists across threads."""
    return CompositeBackend(
        default=StateBackend(),
        routes={MEMORY_PREFIX: StoreBackend(store=store, namespace=lambda _rt: MEMORY_NS)},
    )


@asynccontextmanager
async def open_runtime(state_dir: Path | str) -> AsyncIterator[Runtime]:
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    async with (
        AsyncSqliteSaver.from_conn_string(str(state / "checkpoints.sqlite")) as saver,
        AsyncSqliteStore.from_conn_string(str(state / "store.sqlite")) as store,
    ):
        await store.setup()
        backend = build_backend(store)
        if not (await backend.als("/memories/")).entries:
            await backend.awrite(SEED_MEMORY, SEED_TEXT)
        yield Runtime(saver, store, backend)
