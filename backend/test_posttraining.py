"""Focused tests for deterministic post-training domain logic."""

import pytest

from app.posttraining import (
    Artifact,
    ArtifactKind,
    BenchmarkCase,
    CycleState,
    CycleStateMachine,
    Evidence,
    EvidenceKind,
    EvidenceLabel,
    InvalidTransition,
    PromotionDecision,
    PromotionGate,
    run_benchmark,
)

MANIFEST = "a" * 64
ARTIFACT_HASH = "b" * 64


def artifact(artifact_id: str) -> Artifact:
    return Artifact(
        artifact_id=artifact_id,
        kind=ArtifactKind.CHECKPOINT,
        uri=f"s3://post-training/{artifact_id}",
        sha256=ARTIFACT_HASH,
    )


def evidence(evidence_id: str, artifact_id: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        kind=EvidenceKind.EVALUATION,
        label=EvidenceLabel.PRIOR_VERIFIED_RUN,
        artifact_ids=(artifact_id,),
        metrics={"accuracy": 0.5},
        verified=True,
        benchmark_id="agent-eval-fixed",
        suite="agent-eval",
        suite_version="v1",
        manifest_sha256=MANIFEST,
        seed=17,
    )


def test_promotion_gate_requires_improvement_regression_and_provenance() -> None:
    gate = PromotionGate()
    champion_evidence = evidence("champion-eval", "champion")
    candidate_evidence = evidence("candidate-eval", "candidate")

    result = gate.evaluate(
        0.50,
        0.56,
        candidate_evidence=candidate_evidence,
        champion_evidence=champion_evidence,
    )
    assert result.decision is PromotionDecision.PROMOTE
    assert result.passed is True
    assert result.relative_improvement == pytest.approx(0.12)

    missing_provenance = gate.evaluate(0.50, 0.56)
    assert missing_provenance.decision is PromotionDecision.REJECT
    assert missing_provenance.provenance_passed is False

    regression = gate.evaluate(
        0.50,
        0.56,
        regression_score=0.06,
        candidate_evidence=candidate_evidence,
        champion_evidence=champion_evidence,
    )
    assert regression.decision is PromotionDecision.REJECT
    assert regression.regression_passed is False


def test_state_machine_records_approval_and_rollback() -> None:
    machine = CycleStateMachine.create(
        "cycle-1", champion=artifact("champion"), candidate=artifact("candidate")
    )
    assert machine.state is CycleState.CREATED
    machine.mark_benchmarked()
    machine.mark_evaluated()
    machine.approve()
    machine.rollback("smoke rollback")

    assert machine.state is CycleState.ROLLED_BACK
    assert [(event.from_state, event.to_state) for event in machine.cycle.events] == [
        (CycleState.CREATED, CycleState.BENCHMARKED),
        (CycleState.BENCHMARKED, CycleState.EVALUATED),
        (CycleState.EVALUATED, CycleState.APPROVED),
        (CycleState.APPROVED, CycleState.ROLLED_BACK),
    ]


def test_state_machine_rejects_illegal_transition_and_terminal_reentry() -> None:
    machine = CycleStateMachine.create("cycle-2")
    with pytest.raises(InvalidTransition):
        machine.approve()

    machine.mark_benchmarked()
    machine.mark_evaluated()
    machine.reject("gate failed")
    with pytest.raises(InvalidTransition):
        machine.rollback()


def test_fixed_seed_benchmark_is_reproducible_and_does_not_mutate_cases() -> None:
    cases = [
        BenchmarkCase(case_id="a", input=1, expected=2),
        BenchmarkCase(case_id="b", input=2, expected=4),
        BenchmarkCase(case_id="c", input=3, expected=6),
    ]
    first = run_benchmark(cases, lambda value: value * 2, seed=23)
    second = run_benchmark(cases, lambda value: value * 2, seed=23)

    assert first == second
    assert first.accuracy == 1.0
    assert first.case_order != tuple(case.case_id for case in cases)
    assert tuple(case.case_id for case in cases) == ("a", "b", "c")

