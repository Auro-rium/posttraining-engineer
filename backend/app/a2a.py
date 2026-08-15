"""Typed A2A handoff contracts and an auditable local handoff log."""

from __future__ import annotations

import asyncio
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from .models import (
    AgentRole,
    DatasetManifest,
    DomainModel,
    EvaluationReport,
    FailureCluster,
    Hypothesis,
    PromotionDecision,
    QLoRAConfig,
    TrainingResult,
    new_id,
    utc_now,
)


class A2AArtifactType(StrEnum):
    FAILURE_REPORT = "FailureReport"
    RESEARCH_HYPOTHESIS = "ResearchHypothesis"
    DATASET_MANIFEST = "DatasetManifest"
    EXPERIMENT_SPECIFICATION = "ExperimentSpecification"
    TRAINING_COMPLETED = "TrainingCompleted"
    EVALUATION_REPORT = "EvaluationReport"
    PROMOTION_DECISION = "PromotionDecision"


class FailureReport(DomainModel):
    clusters: list[FailureCluster] = Field(min_length=1)


class ExperimentSpecification(DomainModel):
    hypothesis: Hypothesis
    dataset: DatasetManifest
    config: QLoRAConfig


_ARTIFACT_MODELS: dict[A2AArtifactType, type[DomainModel]] = {
    A2AArtifactType.FAILURE_REPORT: FailureReport,
    A2AArtifactType.RESEARCH_HYPOTHESIS: Hypothesis,
    A2AArtifactType.DATASET_MANIFEST: DatasetManifest,
    A2AArtifactType.EXPERIMENT_SPECIFICATION: ExperimentSpecification,
    A2AArtifactType.TRAINING_COMPLETED: TrainingResult,
    A2AArtifactType.EVALUATION_REPORT: EvaluationReport,
    A2AArtifactType.PROMOTION_DECISION: PromotionDecision,
}


class AgentCard(DomainModel):
    name: str = Field(min_length=1)
    service_url: str = Field(min_length=1)
    roles: list[AgentRole] = Field(min_length=1)
    description: str = Field(min_length=1)
    protocol_version: str = "0.3.0"
    capabilities: list[str] = Field(default_factory=lambda: ["typed-artifacts"])


class A2AEnvelope(DomainModel):
    """Versioned wire envelope whose payload must match ``artifact_type``."""

    message_id: str = Field(default_factory=lambda: new_id("a2a"))
    run_id: str = Field(min_length=1)
    sender: AgentRole
    receiver: AgentRole
    artifact_type: A2AArtifactType
    payload: dict[str, Any]
    schema_version: str = "1.0"
    trace_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_typed_payload(self) -> A2AEnvelope:
        _ARTIFACT_MODELS[self.artifact_type].model_validate(self.payload)
        if self.sender is self.receiver:
            raise ValueError("A2A sender and receiver must differ")
        return self

    def typed_payload(self) -> DomainModel:
        return _ARTIFACT_MODELS[self.artifact_type].model_validate(self.payload)

    @classmethod
    def from_artifact(
        cls,
        *,
        run_id: str,
        sender: AgentRole,
        receiver: AgentRole,
        artifact_type: A2AArtifactType,
        artifact: DomainModel,
        trace_id: str | None = None,
    ) -> A2AEnvelope:
        expected = _ARTIFACT_MODELS[artifact_type]
        if not isinstance(artifact, expected):
            raise TypeError(f"{artifact_type.value} requires {expected.__name__}")
        return cls(
            run_id=run_id,
            sender=sender,
            receiver=receiver,
            artifact_type=artifact_type,
            payload=artifact.model_dump(mode="json"),
            trace_id=trace_id,
        )


class HandoffStatus(StrEnum):
    PENDING = "PENDING"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"


class HandoffRecord(DomainModel):
    envelope: A2AEnvelope
    status: HandoffStatus = HandoffStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    error_code: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class InMemoryHandoffLog:
    """Idempotent audit log used by local A2A delivery and tests."""

    def __init__(self) -> None:
        self._records: dict[str, HandoffRecord] = {}
        self._lock = asyncio.Lock()

    async def record(self, envelope: A2AEnvelope) -> HandoffRecord:
        async with self._lock:
            existing = self._records.get(envelope.message_id)
            if existing is not None:
                return existing.model_copy(deep=True)
            record = HandoffRecord(envelope=envelope)
            self._records[envelope.message_id] = record
            return record.model_copy(deep=True)

    async def mark_delivered(self, message_id: str) -> HandoffRecord:
        return await self._update(message_id, HandoffStatus.DELIVERED)

    async def mark_failed(self, message_id: str, *, error_code: str) -> HandoffRecord:
        return await self._update(message_id, HandoffStatus.FAILED, error_code=error_code)

    async def _update(
        self, message_id: str, status: HandoffStatus, *, error_code: str | None = None
    ) -> HandoffRecord:
        async with self._lock:
            try:
                current = self._records[message_id]
            except KeyError as exc:
                raise KeyError(f"unknown A2A message: {message_id}") from exc
            updated = current.model_copy(
                update={
                    "status": status,
                    "attempts": current.attempts + 1,
                    "error_code": error_code,
                    "updated_at": utc_now(),
                },
                deep=True,
            )
            self._records[message_id] = updated
            return updated.model_copy(deep=True)

    async def list_for_run(self, run_id: str) -> list[HandoffRecord]:
        async with self._lock:
            records = [
                record
                for record in self._records.values()
                if record.envelope.run_id == run_id
            ]
            records.sort(key=lambda item: item.envelope.created_at)
            return [record.model_copy(deep=True) for record in records]
