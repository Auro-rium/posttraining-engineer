from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.a2a import (
    A2AArtifactType,
    A2AEnvelope,
    HandoffStatus,
    InMemoryHandoffLog,
)
from app.artifacts import LocalArtifactStore
from app.models import AgentRole, ArtifactKind, Hypothesis
from app.rag import KnowledgeDocument, KnowledgeScope, LeakageSafeRAG
from app.telemetry import safe_attributes, telemetry_span


def test_rag_rejects_evaluation_splits_and_returns_citations() -> None:
    with pytest.raises((ValidationError, ValueError), match="cannot enter"):
        KnowledgeDocument(
            title="sealed task",
            text="answer for an evaluation task",
            scope=KnowledgeScope.HELDOUT,
        )
    rag = LeakageSafeRAG()
    assert rag.ingest(
        [
            KnowledgeDocument(
                title="QLoRA recipe",
                text="QLoRA uses low rank adapters for efficient post training.",
                scope=KnowledgeScope.DOCUMENTATION,
                source_uri="https://example.test/qlora",
            )
        ],
        chunk_words=20,
    ) == 1
    results = rag.search("low rank QLoRA", limit=1)
    assert results[0].title == "QLoRA recipe"
    assert results[0].source_uri == "https://example.test/qlora"


def test_a2a_envelope_validates_payload_and_handoff_is_idempotent() -> None:
    artifact = Hypothesis(
        failure_cluster_id="cluster-1",
        statement="Preserve constraints in search calls.",
        expected_improvement="Higher held-out success.",
        data_strategy="Replay verified train-side repairs.",
    )
    envelope = A2AEnvelope.from_artifact(
        run_id="RUN-1",
        sender=AgentRole.RESEARCH_AGENT,
        receiver=AgentRole.DATA_CURATOR,
        artifact_type=A2AArtifactType.RESEARCH_HYPOTHESIS,
        artifact=artifact,
    )
    assert envelope.typed_payload() == artifact
    with pytest.raises(ValidationError):
        A2AEnvelope(
            run_id="RUN-1",
            sender=AgentRole.RESEARCH_AGENT,
            receiver=AgentRole.DATA_CURATOR,
            artifact_type=A2AArtifactType.DATASET_MANIFEST,
            payload=artifact.model_dump(mode="json"),
        )

    async def scenario() -> None:
        log = InMemoryHandoffLog()
        first = await log.record(envelope)
        second = await log.record(envelope)
        assert first == second
        delivered = await log.mark_delivered(envelope.message_id)
        assert delivered.status is HandoffStatus.DELIVERED
        assert delivered.attempts == 1

    asyncio.run(scenario())


def test_local_artifacts_are_integrity_checked(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = LocalArtifactStore(tmp_path)
        ref = await store.put_json(
            key="runs/RUN-1/report.json",
            value={"success": 0.4},
            kind=ArtifactKind.EVALUATION_REPORT,
        )
        assert await store.exists(ref)
        assert await store.get_json(ref) == {"success": 0.4}
        with pytest.raises(ValueError, match="relative path"):
            await store.put_bytes(
                key="../escape",
                data=b"unsafe",
                kind=ArtifactKind.DATASET,
                content_type="application/octet-stream",
            )

    asyncio.run(scenario())


def test_telemetry_drops_content_and_secret_attributes() -> None:
    cleaned = safe_attributes(
        {
            "run_id": "RUN-1",
            "token_count": 10,
            "prompt_content": "do not export",
            "api_key": "do not export",
            "unknown": "do not export",
        }
    )
    assert cleaned == {"run_id": "RUN-1", "token_count": 10}
    with telemetry_span("test", attributes={"run_id": "RUN-1"}) as span:
        assert span is not None
