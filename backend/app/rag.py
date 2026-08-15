"""Small leakage-safe retrieval layer used by the research agents.

This lexical implementation keeps the backend runnable without an embedding
service.  It exposes the same narrow interface a production vector index can
implement later, while enforcing the evaluation-data boundary at ingestion and
retrieval time.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum

from pydantic import Field, model_validator

from .models import Citation, DomainModel, new_id


class KnowledgeScope(StrEnum):
    DOCUMENTATION = "documentation"
    TRAIN = "train"
    EXPERIMENT = "experiment"
    HELDOUT = "heldout"
    REGRESSION = "regression"


FORBIDDEN_SCOPES = {KnowledgeScope.HELDOUT, KnowledgeScope.REGRESSION}


class LeakageBoundaryError(ValueError):
    """Raised when evaluation material is offered to the research index."""


class KnowledgeDocument(DomainModel):
    document_id: str = Field(default_factory=lambda: new_id("document"))
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    scope: KnowledgeScope
    source_uri: str | None = None
    task_id: str | None = None

    @model_validator(mode="after")
    def reject_heldout_material(self) -> KnowledgeDocument:
        if self.scope in FORBIDDEN_SCOPES:
            raise LeakageBoundaryError(
                f"{self.scope.value} material cannot enter the research corpus"
            )
        return self


class KnowledgeChunk(DomainModel):
    document_id: str
    chunk_id: str
    title: str
    text: str
    scope: KnowledgeScope
    source_uri: str | None = None


def _tokens(text: str) -> list[str]:
    """Normalize text into stable ASCII word tokens for local retrieval."""

    return re.findall(r"[a-z0-9_]+", text.lower())


def chunk_document(document: KnowledgeDocument, *, chunk_words: int = 180) -> list[KnowledgeChunk]:
    """Split a permitted document into deterministic, non-overlapping chunks."""

    if chunk_words < 20:
        raise ValueError("chunk_words must be at least 20")
    # Validate again to protect callers that construct objects without validation.
    if document.scope in FORBIDDEN_SCOPES:
        raise LeakageBoundaryError("evaluation material cannot be chunked for RAG")
    words = document.text.split()
    return [
        KnowledgeChunk(
            document_id=document.document_id,
            chunk_id=f"{document.document_id}:{offset // chunk_words}",
            title=document.title,
            text=" ".join(words[offset : offset + chunk_words]),
            scope=document.scope,
            source_uri=document.source_uri,
        )
        for offset in range(0, len(words), chunk_words)
    ]


class LeakageSafeRAG:
    """In-memory research corpus that fails closed on held-out material."""

    def __init__(self) -> None:
        self._chunks: dict[str, KnowledgeChunk] = {}

    def ingest(self, documents: Iterable[KnowledgeDocument], *, chunk_words: int = 180) -> int:
        pending: list[KnowledgeChunk] = []
        for document in documents:
            if document.scope in FORBIDDEN_SCOPES:
                raise LeakageBoundaryError(
                    f"refusing to index {document.scope.value} document {document.document_id}"
                )
            pending.extend(chunk_document(document, chunk_words=chunk_words))
        for chunk in pending:
            self._chunks[chunk.chunk_id] = chunk
        return len(pending)

    def search(self, query: str, *, limit: int = 5) -> list[Citation]:
        if not query.strip() or limit < 1:
            return []
        query_terms = Counter(_tokens(query))
        if not query_terms:
            return []
        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk in self._chunks.values():
            if chunk.scope in FORBIDDEN_SCOPES:
                # A second boundary protects against a corrupted backing index.
                continue
            chunk_terms = Counter(_tokens(f"{chunk.title} {chunk.text}"))
            overlap = sum(
                min(query_count, chunk_terms.get(term, 0))
                for term, query_count in query_terms.items()
            )
            if overlap:
                score = overlap / math.sqrt(sum(query_terms.values()) * sum(chunk_terms.values()))
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            Citation(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                source_uri=chunk.source_uri,
                excerpt=chunk.text[:1200],
                score=score,
            )
            for score, chunk in scored[:limit]
        ]

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)
