"""Focused contract tests for the bounded run-history registry."""

from datetime import UTC, datetime

import pytest

from app.posttraining.models import Artifact, ArtifactKind, Evidence, EvidenceKind, EvidenceLabel
from app.posttraining.run_history import (
    MAX_RUNS,
    BenchmarkMetrics,
    ComparisonDTO,
    RunDecision,
    RunHistoryRecord,
    RunHistoryRepository,
    RunLimitExceeded,
    RunRegistry,
    RunStatus,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def checkpoint(artifact_id: str, digest: str = HASH_A) -> Artifact:
    return Artifact(
        artifact_id=artifact_id,
        kind=ArtifactKind.CHECKPOINT,
        uri=f"s3://demo-artifacts/{artifact_id}",
        sha256=digest,
    )


def evaluation_evidence(number: int) -> Evidence:
    return Evidence(
        evidence_id=f"evaluation-{number:03d}",
        kind=EvidenceKind.EVALUATION,
        label=EvidenceLabel.LIVE,
        artifact_ids=(f"candidate-{number:03d}",),
        metrics={"aggregate": 0.50 + number / 100, "WebShop": 0.45, "Wordle": 0.50},
        verified=True,
        benchmark_id="agentgym-held-out",
        suite="AgentGym",
        suite_version="v1",
        manifest_sha256=HASH_A,
        seed=7,
        model_id="google/functiongemma-270m-it",
    )


def run_record(
    number: int,
    *,
    status: RunStatus = RunStatus.COMPLETED,
    decision: RunDecision | None = RunDecision.PROMOTE,
    parent_run_id: str | None = None,
    champion_run_id: str | None = None,
    manifest_sha256: str | None = HASH_A,
) -> RunHistoryRecord:
    return RunHistoryRecord(
        run_id=f"run-{number:03d}",
        run_number=number,
        parent_run_id=parent_run_id,
        champion_run_id=champion_run_id,
        champion_artifact_id="champion-checkpoint",
        candidate_artifact_id=f"candidate-{number:03d}",
        status=status,
        decision=decision,
        manifest_sha256=manifest_sha256,
        benchmark_id="agentgym-held-out",
        suite="AgentGym",
        suite_version="v1",
        seed=7,
        model_id="google/functiongemma-270m-it",
        baseline_metrics=BenchmarkMetrics(
            aggregate=0.40 + number / 100,
            per_environment={"WebShop": 0.30, "Wordle": 0.40},
        ),
        candidate_metrics=BenchmarkMetrics(
            aggregate=0.50 + number / 100,
            per_environment={"WebShop": 0.45, "Wordle": 0.50},
        ),
        artifact_refs=(checkpoint(f"candidate-{number:03d}", HASH_B),),
        evidence=(evaluation_evidence(number),),
        created_at=datetime(2026, 9, number, tzinfo=UTC),
        updated_at=datetime(2026, 9, number, tzinfo=UTC),
    )


class MemoryRunHistory:
    """Minimal repository double that implements the public contract."""

    def __init__(self) -> None:
        self.records: dict[str, RunHistoryRecord] = {}

    def reserve_run(
        self, record: RunHistoryRecord, *, max_runs: int = MAX_RUNS
    ) -> RunHistoryRecord:
        if record.run_id in self.records:
            raise ValueError("duplicate run id")
        if len(self.records) >= max_runs:
            raise RunLimitExceeded(f"maximum of {max_runs} runs reached")
        self.records[record.run_id] = record
        return record

    def get_run(self, run_id: str) -> RunHistoryRecord | None:
        return self.records.get(run_id)

    def get_history_run(self, run_id: str) -> RunHistoryRecord | None:
        return self.records.get(run_id)

    def list_runs(self, *, limit: int = MAX_RUNS) -> tuple[RunHistoryRecord, ...]:
        return tuple(sorted(self.records.values(), key=lambda record: record.run_number)[-limit:])

    def list_history_runs(self, *, limit: int = MAX_RUNS) -> tuple[RunHistoryRecord, ...]:
        return self.list_runs(limit=limit)


def test_run_record_validates_identity_and_metrics() -> None:
    record = run_record(1)

    assert record.run_id == "run-001"
    assert record.run_number == 1
    candidate_metrics = record.candidate_metrics
    assert candidate_metrics is not None
    assert candidate_metrics.aggregate == pytest.approx(0.51)
    assert candidate_metrics.per_environment["WebShop"] == pytest.approx(0.45)
    assert record.artifact_refs[0].sha256 == HASH_B

    with pytest.raises(ValueError, match="run_number"):
        RunHistoryRecord(
            run_id="run-006",
            run_number=6,
        )


def test_registry_reserves_at_most_five_runs_and_preserves_links() -> None:
    repository: RunHistoryRepository = MemoryRunHistory()
    registry = RunRegistry(repository)

    for number in range(1, MAX_RUNS + 1):
        parent = None if number == 1 else f"run-{number - 1:03d}"
        registry.register(
            run_record(
                number,
                parent_run_id=parent,
                champion_run_id=parent,
            )
        )

    assert len(registry.list_runs()) == MAX_RUNS
    assert registry.get("run-005").parent_run_id == "run-004"

    with pytest.raises(RunLimitExceeded, match="maximum of 5"):
        registry.register(
            run_record(5, parent_run_id="run-004", champion_run_id="run-004").model_copy(
                update={"run_id": "run-overflow"}
            )
        )


def test_registry_builds_api_and_graph_comparison_dto() -> None:
    repository: RunHistoryRepository = MemoryRunHistory()
    registry = RunRegistry(repository)
    registry.register(
        run_record(
            1,
            status=RunStatus.REJECTED,
            decision=RunDecision.REJECT,
            manifest_sha256=HASH_A,
        )
    )
    registry.register(run_record(2, parent_run_id="run-001", champion_run_id="run-001"))

    comparison = registry.compare()

    assert isinstance(comparison, ComparisonDTO)
    assert comparison.run_count == 2
    assert comparison.run_ids == ("run-001", "run-002")
    assert comparison.rows[0].decision is RunDecision.REJECT
    assert comparison.rows[0].baseline_aggregate == pytest.approx(0.41)
    assert comparison.rows[0].candidate_aggregate == pytest.approx(0.51)
    assert comparison.rows[0].relative_improvement == pytest.approx(0.243902439)
    assert comparison.rows[1].parent_run_id == "run-001"
    assert comparison.rows[1].candidate_artifact_id == "candidate-002"
    assert comparison.rows[0].candidate_per_environment["Wordle"] == pytest.approx(0.50)


def test_comparison_rejects_unknown_ids_and_more_than_five_ids() -> None:
    repository: RunHistoryRepository = MemoryRunHistory()
    registry = RunRegistry(repository)
    registry.register(run_record(1))

    with pytest.raises(KeyError, match="run-404"):
        registry.compare(("run-404",))

    with pytest.raises(ValueError, match="at most 5"):
        registry.compare(tuple(f"run-{number:03d}" for number in range(1, 7)))


def test_repository_contract_is_runtime_checkable_by_shape() -> None:
    repository = MemoryRunHistory()

    assert isinstance(repository, RunHistoryRepository)


def test_record_requires_manifest_for_terminal_decision() -> None:
    with pytest.raises(ValueError, match="manifest_sha256"):
        run_record(1, manifest_sha256=None)

    pending = run_record(
        1,
        status=RunStatus.RUNNING,
        decision=None,
        manifest_sha256=None,
    )
    assert pending.status is RunStatus.RUNNING
    assert pending.decision is None


def test_record_rejects_mutation_of_nested_metrics_and_artifact_metadata() -> None:
    record = run_record(1)
    assert record.candidate_metrics is not None

    with pytest.raises(TypeError):
        record.candidate_metrics.per_environment["WebShop"] = 0.99
    with pytest.raises(TypeError):
        record.artifact_refs[0].metadata["source"] = "changed"
    with pytest.raises(ValueError, match="frozen"):
        record.evidence[0].verified = False
    with pytest.raises(TypeError):
        record.evidence[0].metrics["aggregate"] = 0.01


def test_terminal_record_requires_verified_provenance_and_artifacts() -> None:
    required = run_record(1).model_dump(mode="python")
    required["evidence"] = ()
    with pytest.raises(ValueError, match="verified evidence"):
        RunHistoryRecord(**required)

    required = run_record(1).model_dump(mode="python")
    required["artifact_refs"] = ()
    with pytest.raises(ValueError, match="artifact references"):
        RunHistoryRecord(**required)

    required = run_record(1).model_dump(mode="python")
    required["benchmark_id"] = None
    with pytest.raises(ValueError, match="benchmark_id"):
        RunHistoryRecord(**required)


@pytest.mark.parametrize(
    ("status", "decision"),
    [
        (RunStatus.COMPLETED, None),
        (RunStatus.REJECTED, None),
        (RunStatus.RUNNING, RunDecision.PROMOTE),
        (RunStatus.COMPLETED, RunDecision.REJECT),
        (RunStatus.REJECTED, RunDecision.PROMOTE),
    ],
)
def test_record_rejects_inconsistent_status_and_decision(
    status: RunStatus, decision: RunDecision | None
) -> None:
    with pytest.raises(ValueError, match="status and decision"):
        run_record(1, status=status, decision=decision)


def test_registry_requires_sequential_existing_links() -> None:
    registry = RunRegistry(MemoryRunHistory())

    with pytest.raises(ValueError, match="next sequential"):
        registry.register(run_record(2, parent_run_id="run-001", champion_run_id="run-001"))

    registry.register(run_record(1))
    with pytest.raises(ValueError, match="parent_run_id"):
        registry.register(run_record(2, parent_run_id="missing", champion_run_id="run-001"))
