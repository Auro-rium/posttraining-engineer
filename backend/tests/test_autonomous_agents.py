"""Contract tests for bounded Nemotron reasoning handoffs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.prompt_contract import NEMOTRON_MODEL_ID
from app.autonomous.agents import (
    AutonomousAgentAdapters,
    CuratedDatasetPlan,
    DuplicateHypothesisError,
    FailureCluster,
    ProviderHandoffError,
    QLoRAConfig,
    ResearchHypothesis,
    validate_qlora_config,
)


class RecordingProvider:
    model_id = NEMOTRON_MODEL_ID

    def __init__(self, response: object) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str, **_: object) -> object:
        self.prompts.append(prompt)
        return self.response


class HistoryRecord:
    status = "rejected"
    hypothesis_id = "h-old"

    def model_dump(self, **_: object) -> dict[str, object]:
        return {"status": self.status, "hypothesis_id": self.hypothesis_id}


def test_handoff_models_reject_missing_evidence_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        FailureCluster(cluster_id="c1", failure_type="bad_tool", count=1)

    with pytest.raises(ValidationError):
        ResearchHypothesis(
            hypothesis_id="h1",
            cluster_id="c1",
            statement="Use the verifier",
            prediction="success improves",
            falsifier="success does not improve",
            evidence_refs=["traj://1"],
            unexpected="nope",
        )

    with pytest.raises(ValidationError):
        CuratedDatasetPlan(
            plan_id="p1",
            selected_trajectory_refs=[],
            dataset_artifact_ref="s3://dataset",
        )


def test_failure_analysis_prompt_contains_all_prior_experiment_evidence() -> None:
    provider = RecordingProvider(
        {
            "clusters": [
                {
                    "cluster_id": "cluster-1",
                    "failure_type": "premature_completion",
                    "description": "Completion happened before healthcheck",
                    "count": 2,
                    "evidence_refs": ["traj://failed-1"],
                }
            ]
        }
    )
    adapters = AutonomousAgentAdapters(provider)
    history = [{"experiment_id": "exp-1", "status": "rejected", "evidence_refs": ["s3://eval-1"]}]

    clusters = adapters.analyze_failures(
        trajectory_references=["traj://failed-1"], experiment_history=history
    )

    assert clusters[0].cluster_id == "cluster-1"
    assert "traj://failed-1" in provider.prompts[0]
    assert "s3://eval-1" in provider.prompts[0]
    assert '"experiment_history"' in provider.prompts[0]


def test_handoff_accepts_pydantic_style_experiment_history_records() -> None:
    provider = RecordingProvider(
        {
            "clusters": [
                {
                    "cluster_id": "cluster-1",
                    "failure_type": "premature_completion",
                    "description": "Completion happened before healthcheck",
                    "count": 1,
                    "evidence_refs": ["traj://failed-1"],
                }
            ]
        }
    )

    adapters = AutonomousAgentAdapters(provider)
    adapters.analyze_failures(["traj://failed-1"], experiment_history=[HistoryRecord()])

    assert '"hypothesis_id":"h-old"' in provider.prompts[0]


def test_research_rejects_a_hypothesis_that_previously_failed() -> None:
    provider = RecordingProvider(
        {
            "hypotheses": [
                {
                    "hypothesis_id": "h-new",
                    "cluster_id": "cluster-1",
                    "statement": "Verify after restart",
                    "prediction": "fewer premature completions",
                    "falsifier": "premature completions do not decrease",
                    "evidence_refs": ["traj://failed-1"],
                }
            ]
        }
    )
    adapters = AutonomousAgentAdapters(provider)
    clusters = [
        FailureCluster(
            cluster_id="cluster-1",
            failure_type="premature_completion",
            description="Completion happened before healthcheck",
            count=2,
            evidence_refs=["traj://failed-1"],
        )
    ]
    history = [
        {
            "hypothesis_id": "h-old",
            "cluster_id": "cluster-1",
            "statement": "Verify after restart",
            "prediction": "fewer premature completions",
            "falsifier": "premature completions do not decrease",
            "evidence_refs": ["traj://failed-1"],
            "status": "failed",
        }
    ]

    with pytest.raises(DuplicateHypothesisError):
        adapters.research(failure_clusters=clusters, experiment_history=history)


def test_provider_failure_is_not_converted_to_a_fabricated_handoff() -> None:
    class BrokenProvider:
        model_id = NEMOTRON_MODEL_ID

        def invoke(self, *_: object, **__: object) -> object:
            raise TimeoutError("provider unavailable")

    adapters = AutonomousAgentAdapters(BrokenProvider())

    with pytest.raises(ProviderHandoffError, match="provider unavailable"):
        adapters.analyze_failures(trajectory_references=["traj://1"], experiment_history=[])


def test_strict_json_rejects_unknown_wrapper_keys() -> None:
    provider = RecordingProvider(
        {
            "clusters": [
                {
                    "cluster_id": "cluster-1",
                    "failure_type": "premature_completion",
                    "description": "Completion happened before healthcheck",
                    "count": 1,
                    "evidence_refs": ["traj://failed-1"],
                }
            ],
            "unexpected": "must be rejected",
        }
    )

    with pytest.raises(ProviderHandoffError, match="unknown keys"):
        AutonomousAgentAdapters(provider).analyze_failures(["traj://failed-1"], [])


def test_qlora_validator_enforces_the_exact_bounded_search_space() -> None:
    valid = {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "learning_rate": 2e-4,
        "epochs": 2,
        "sequence_length": 1024,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    assert validate_qlora_config(valid).rank == 16
    assert QLoRAConfig.model_validate(valid).target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )

    with pytest.raises(ValidationError):
        validate_qlora_config({**valid, "rank": 64})
    with pytest.raises(ValidationError):
        validate_qlora_config({**valid, "target_modules": ["q_proj", "k_proj"]})


def test_handoff_models_accept_only_documented_legacy_field_aliases() -> None:
    cluster = FailureCluster(
        cluster_id="cluster-1",
        failure_type="premature_completion",
        description="Completion happened before healthcheck",
        example_count=2,
        example_trajectories=["traj://failed-1"],
    )
    hypothesis = ResearchHypothesis(
        hypothesis_id="h1",
        cluster_id="cluster-1",
        statement="Verify after restart",
        testable_prediction="fewer premature completions",
        falsifiable_criterion="premature completions do not decrease",
        evidence_references=["traj://failed-1"],
    )
    plan = CuratedDatasetPlan(
        dataset_plan_id="p1",
        selected_record_ids=["traj://failed-1"],
        dataset_reference="s3://dataset",
    )

    assert cluster.count == 2
    assert hypothesis.prediction == "fewer premature completions"
    assert plan.selected_trajectory_refs == ("traj://failed-1",)
