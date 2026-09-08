"""Bounded, typed handoffs for the autonomous live run.

Nemotron may suggest a failure explanation, a falsifiable experiment, a
dataset selection, or a QLoRA configuration.  It is never allowed to invent
provider evidence.  This module therefore keeps the model boundary narrow:
the provider returns strict JSON and deterministic Python validation owns the
handoff contracts and QLoRA search space.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from app.agents.prompt_contract import NEMOTRON_MODEL_ID, get_prompt_contract


class ProviderHandoffError(RuntimeError):
    """A provider response cannot safely become an autonomous handoff."""


class DuplicateHypothesisError(ValueError):
    """A previously failed hypothesis was proposed without new evidence."""


class NemotronProvider(Protocol):
    """Small provider protocol accepted by the adapter.

    ``BedrockStrandsModel`` implements ``invoke`` and tests can inject a
    deterministic provider with the same method.  The adapter never supplies
    a fallback model when this call fails.
    """

    model_id: str

    def invoke(self, prompt: str, *, agent_name: str, system_prompt: str) -> Any:
        """Return the provider's raw response."""


_EVIDENCE_LABELS: Final[frozenset[str]] = frozenset({"LIVE", "PRIOR_VERIFIED_RUN", "EXPLANATION"})
_VERIFIED_EVIDENCE_LABELS: Final[frozenset[str]] = frozenset({"LIVE", "PRIOR_VERIFIED_RUN"})
_QLORA_SEARCH_SPACE: Final[dict[str, tuple[Any, ...]]] = {
    "rank": (8, 16, 32),
    "alpha": (16, 32, 64),
    "dropout": (0.0, 0.05, 0.1),
    "learning_rate": (1e-4, 2e-4, 5e-4),
    "epochs": (1, 2, 3),
    "sequence_length": (512, 1024),
    "batch_size": (1, 2, 4),
    "gradient_accumulation_steps": (4, 8, 16),
    "target_modules": (("q_proj", "k_proj", "v_proj", "o_proj"),),
}
QLORA_SEARCH_SPACE: Final[dict[str, tuple[Any, ...]]] = _QLORA_SEARCH_SPACE
_TARGET_MODULES: Final[tuple[str, ...]] = ("q_proj", "k_proj", "v_proj", "o_proj")


class _HandoffModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FailureCluster(_HandoffModel):
    """One verified, train-side failure pattern."""

    cluster_id: str = Field(min_length=1)
    failure_type: str = Field(min_length=1)
    description: str = Field(min_length=1)
    count: int = Field(
        ge=1,
        validation_alias=AliasChoices("count", "example_count"),
    )
    evidence_refs: tuple[str, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "evidence_refs", "trajectory_refs", "evidence_references", "example_trajectories"
        ),
    )
    evidence_class: str = "LIVE"

    @model_validator(mode="after")
    def validate_evidence(self) -> FailureCluster:
        if self.evidence_class not in _EVIDENCE_LABELS:
            raise ValueError("evidence_class must be LIVE, PRIOR_VERIFIED_RUN, or EXPLANATION")
        if self.evidence_class not in _VERIFIED_EVIDENCE_LABELS:
            raise ValueError("failure clusters require verified evidence")
        if any(not ref.strip() for ref in self.evidence_refs):
            raise ValueError("evidence_refs must contain non-empty references")
        return self

    @property
    def trajectory_refs(self) -> tuple[str, ...]:
        """Compatibility name used by trajectory-oriented callers."""

        return self.evidence_refs


class ResearchHypothesis(_HandoffModel):
    """A falsifiable repair hypothesis grounded in observed failures."""

    hypothesis_id: str = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    prediction: str = Field(
        min_length=1,
        validation_alias=AliasChoices("prediction", "testable_prediction"),
    )
    falsifier: str = Field(
        min_length=1,
        validation_alias=AliasChoices("falsifier", "falsifiable_criterion"),
    )
    evidence_refs: tuple[str, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices("evidence_refs", "evidence_references"),
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_class: str = "EXPLANATION"

    @model_validator(mode="after")
    def validate_evidence(self) -> ResearchHypothesis:
        if self.evidence_class not in _EVIDENCE_LABELS:
            raise ValueError("evidence_class must be LIVE, PRIOR_VERIFIED_RUN, or EXPLANATION")
        if not self.evidence_refs or any(not ref.strip() for ref in self.evidence_refs):
            raise ValueError("evidence_refs must contain non-empty references")
        return self

    @property
    def testable_prediction(self) -> str:
        return self.prediction

    @property
    def falsifiable_criterion(self) -> str:
        return self.falsifier

    @property
    def fingerprint(self) -> str:
        """Stable content identity used for failed-hypothesis de-duplication."""

        return hypothesis_fingerprint(self)


class CuratedDatasetPlan(_HandoffModel):
    """Selection of verified trajectory references eligible for SFT."""

    plan_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("plan_id", "dataset_plan_id"),
    )
    selected_trajectory_refs: tuple[str, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "selected_trajectory_refs", "trajectory_references", "selected_record_ids"
        ),
    )
    dataset_artifact_ref: str = Field(
        min_length=1,
        validation_alias=AliasChoices("dataset_artifact_ref", "dataset_reference"),
    )
    record_count: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("record_count", "selected_record_count"),
    )
    evidence_class: str = "LIVE"

    @model_validator(mode="after")
    def validate_evidence(self) -> CuratedDatasetPlan:
        if self.evidence_class not in _VERIFIED_EVIDENCE_LABELS:
            raise ValueError("curated datasets require LIVE or PRIOR_VERIFIED_RUN evidence")
        if any(not ref.strip() for ref in self.selected_trajectory_refs):
            raise ValueError("selected_trajectory_refs must contain non-empty references")
        if self.record_count is not None and self.record_count != len(
            self.selected_trajectory_refs
        ):
            raise ValueError("record_count must equal selected_trajectory_refs length")
        return self

    @property
    def selected_record_ids(self) -> tuple[str, ...]:
        return self.selected_trajectory_refs


class QLoRAConfig(_HandoffModel):
    """The only trainable configuration admitted by the live workflow."""

    rank: int
    alpha: int
    dropout: float
    learning_rate: float
    epochs: int
    sequence_length: int
    batch_size: int
    gradient_accumulation_steps: int
    target_modules: tuple[str, ...]

    @model_validator(mode="after")
    def validate_search_space(self) -> QLoRAConfig:
        values = self.model_dump()
        for name, allowed in _QLORA_SEARCH_SPACE.items():
            actual = values[name]
            if name == "target_modules":
                actual = tuple(actual)
            if actual not in allowed:
                raise ValueError(f"{name}={actual!r} is outside the fixed QLoRA search space")
        if tuple(self.target_modules) != _TARGET_MODULES:
            raise ValueError("target_modules must be q_proj,k_proj,v_proj,o_proj in that order")
        return self


def validate_qlora_config(config: QLoRAConfig | Mapping[str, Any]) -> QLoRAConfig:
    """Parse and deterministically validate one exact-search-space config."""

    if isinstance(config, QLoRAConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("QLoRA configuration must be a mapping")
    return QLoRAConfig.model_validate(dict(config))


def hypothesis_fingerprint(hypothesis: ResearchHypothesis | Mapping[str, Any]) -> str:
    """Hash hypothesis meaning, excluding its provider-generated ID."""

    if isinstance(hypothesis, ResearchHypothesis):
        values = hypothesis.model_dump(
            include={"cluster_id", "statement", "prediction", "falsifier"}
        )
    elif isinstance(hypothesis, Mapping):
        values = {
            key: hypothesis.get(key)
            for key in ("cluster_id", "statement", "prediction", "falsifier")
        }
        if values["prediction"] is None:
            values["prediction"] = hypothesis.get("testable_prediction")
        if values["falsifier"] is None:
            values["falsifier"] = hypothesis.get("falsifiable_criterion")
    else:
        raise TypeError("hypothesis must be a ResearchHypothesis or mapping")
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_response(raw: Any) -> Any:
    """Decode a provider response without accepting markdown or partial JSON."""

    if isinstance(raw, (Mapping, list)):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        for attribute in ("text", "output", "content"):
            value = getattr(raw, attribute, None)
            if value is not None:
                return _json_response(value)
        raise ProviderHandoffError("provider returned a non-JSON response")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProviderHandoffError("provider returned invalid strict JSON") from exc


def _mapping_history(history: Sequence[Any] | None) -> list[dict[str, Any]]:
    if history is None:
        return []
    if isinstance(history, (str, bytes)):
        raise TypeError("experiment_history must be a sequence of mappings")
    result: list[dict[str, Any]] = []
    for item in history:
        if isinstance(item, Mapping):
            result.append(dict(item))
            continue
        dump = getattr(item, "model_dump", None)
        if callable(dump):
            dumped = dump(mode="json")
            if isinstance(dumped, Mapping):
                result.append(dict(dumped))
                continue
        raise TypeError("experiment_history entries must be mappings or Pydantic records")
    return result


def _collection(response: Any, key: str) -> list[Any]:
    decoded = _json_response(response)
    if isinstance(decoded, list):
        values = decoded
    elif isinstance(decoded, Mapping):
        status = decoded.get("status")
        if status in {"BLOCKED", "FAILED", "ERROR", "blocked", "failed", "error"}:
            raise ProviderHandoffError(f"provider handoff returned {status}")
        unknown = set(decoded).difference({"status", key})
        if unknown:
            raise ProviderHandoffError(
                f"provider response contains unknown keys: {sorted(unknown)}"
            )
        candidate = decoded.get(key)
        if not isinstance(candidate, list):
            raise ProviderHandoffError(f"provider response must contain a JSON list named {key}")
        values = candidate
    else:
        raise ProviderHandoffError("provider response must be a JSON object or list")
    return values


def _object_response(response: Any, key: str) -> dict[str, Any]:
    decoded = _json_response(response)
    if not isinstance(decoded, Mapping):
        raise ProviderHandoffError("provider response must be a JSON object")
    status = decoded.get("status")
    if status in {"BLOCKED", "FAILED", "ERROR", "blocked", "failed", "error"}:
        raise ProviderHandoffError(f"provider handoff returned {status}")
    if key in decoded:
        unknown = set(decoded).difference({"status", key})
        if unknown:
            raise ProviderHandoffError(
                f"provider response contains unknown keys: {sorted(unknown)}"
            )
        value = decoded[key]
    else:
        value = decoded
    if not isinstance(value, Mapping):
        raise ProviderHandoffError(f"provider response field {key} must be a JSON object")
    return dict(value)


@dataclass(frozen=True)
class AutonomousAgentAdapters:
    """Four bounded Nemotron handoffs used by the autonomous supervisor."""

    provider: NemotronProvider

    def __post_init__(self) -> None:
        provider_model_id = getattr(self.provider, "model_id", None)
        if provider_model_id is not None and provider_model_id != NEMOTRON_MODEL_ID:
            raise ValueError(
                "autonomous reasoning requires pinned Nemotron model "
                f"{NEMOTRON_MODEL_ID}; received {provider_model_id!r}"
            )
        if not callable(getattr(self.provider, "invoke", None)) and not callable(self.provider):
            raise TypeError("provider must expose invoke(prompt, ...) or be callable")

    def _call(self, agent_key: str, payload: Mapping[str, Any]) -> Any:
        contract = get_prompt_contract(agent_key)
        prompt = contract.render_handoff(payload)
        target = getattr(self.provider, "invoke", None)
        if target is None and callable(self.provider):
            target = self.provider
        if target is None:  # pragma: no cover - guarded in __post_init__
            raise ProviderHandoffError("provider has no invocation method")
        try:
            try:
                return target(
                    prompt,
                    agent_name=agent_key,
                    system_prompt=contract.prompt,
                )
            except TypeError:
                # A minimal callable test adapter may accept only the prompt.
                return target(prompt)
        except Exception as exc:
            raise ProviderHandoffError(f"provider invocation failed: {exc}") from exc

    def analyze_failures(
        self,
        trajectory_references: Sequence[str],
        experiment_history: Sequence[Any] = (),
    ) -> tuple[FailureCluster, ...]:
        refs = _references(trajectory_references, "trajectory_references")
        history = _mapping_history(experiment_history)
        response = self._call(
            "FailureAnalystAgent",
            {"trajectory_references": refs, "experiment_history": history},
        )
        try:
            return tuple(
                FailureCluster.model_validate(item) for item in _collection(response, "clusters")
            )
        except ProviderHandoffError:
            raise
        except Exception as exc:
            raise ProviderHandoffError(
                "provider failure-analysis JSON failed schema validation"
            ) from exc

    failure_analysis = analyze_failures

    def research(
        self,
        failure_clusters: Sequence[FailureCluster | Mapping[str, Any]],
        experiment_history: Sequence[Any] = (),
    ) -> tuple[ResearchHypothesis, ...]:
        clusters = [
            item.model_dump(mode="json") if isinstance(item, FailureCluster) else dict(item)
            for item in failure_clusters
        ]
        history = _mapping_history(experiment_history)
        response = self._call(
            "ResearchAgent",
            {"failure_clusters": clusters, "experiment_history": history},
        )
        try:
            hypotheses = tuple(
                ResearchHypothesis.model_validate(item)
                for item in _collection(response, "hypotheses")
            )
        except ProviderHandoffError:
            raise
        except Exception as exc:
            raise ProviderHandoffError("provider research JSON failed schema validation") from exc
        self._reject_failed_duplicates(hypotheses, history)
        return hypotheses

    research_hypotheses = research

    def curate(
        self,
        verified_trajectory_references: Sequence[str],
        hypotheses: Sequence[ResearchHypothesis | Mapping[str, Any]] = (),
        experiment_history: Sequence[Any] = (),
    ) -> CuratedDatasetPlan:
        refs = _references(verified_trajectory_references, "verified_trajectory_references")
        history = _mapping_history(experiment_history)
        hypothesis_values = [
            item.model_dump(mode="json") if isinstance(item, ResearchHypothesis) else dict(item)
            for item in hypotheses
        ]
        response = self._call(
            "DataCuratorAgent",
            {
                "verified_trajectory_references": refs,
                "hypotheses": hypothesis_values,
                "experiment_history": history,
            },
        )
        try:
            plan = CuratedDatasetPlan.model_validate(_object_response(response, "plan"))
        except ProviderHandoffError:
            raise
        except Exception as exc:
            raise ProviderHandoffError("provider curation JSON failed schema validation") from exc
        if not set(plan.selected_trajectory_refs).issubset(set(refs)):
            raise ProviderHandoffError(
                "curation selected a trajectory outside verified input references"
            )
        return plan

    curate_dataset = curate

    def design_qlora(
        self,
        dataset_plan: CuratedDatasetPlan | Mapping[str, Any],
        experiment_history: Sequence[Any] = (),
    ) -> QLoRAConfig:
        plan = (
            dataset_plan.model_dump(mode="json")
            if isinstance(dataset_plan, CuratedDatasetPlan)
            else dict(dataset_plan)
        )
        history = _mapping_history(experiment_history)
        response = self._call(
            "TrainingDesignerAgent",
            {"dataset_plan": plan, "experiment_history": history},
        )
        try:
            return validate_qlora_config(_object_response(response, "config"))
        except ProviderHandoffError:
            raise
        except Exception as exc:
            raise ProviderHandoffError(
                "provider QLoRA JSON failed bounded schema validation"
            ) from exc

    design_training = design_qlora

    @staticmethod
    def _reject_failed_duplicates(
        hypotheses: Sequence[ResearchHypothesis], history: Sequence[Mapping[str, Any]]
    ) -> None:
        failed = [
            item
            for item in history
            if str(item.get("status", "")).lower() in {"failed", "rejected"}
        ]
        prior: dict[str, set[str]] = {}
        for item in failed:
            fingerprint = item.get("fingerprint")
            if isinstance(fingerprint, str):
                prior.setdefault(fingerprint, set()).update(_history_refs(item))
            try:
                prior.setdefault(hypothesis_fingerprint(item), set()).update(_history_refs(item))
            except (TypeError, ValueError):
                continue
        for hypothesis in hypotheses:
            previous_refs = prior.get(hypothesis.fingerprint)
            if previous_refs is None:
                continue
            current_refs = set(hypothesis.evidence_refs)
            if not current_refs.difference(previous_refs):
                raise DuplicateHypothesisError(
                    f"hypothesis {hypothesis.hypothesis_id!r} repeats a failed hypothesis"
                )


def _history_refs(item: Mapping[str, Any]) -> set[str]:
    for key in ("evidence_refs", "evidence_references", "trajectory_refs", "evidence_refs_used"):
        value = item.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return {str(ref) for ref in value}
    return set()


def _references(value: Sequence[str], name: str) -> list[str]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of opaque references")
    refs = list(value)
    if not refs or any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        raise ValueError(f"{name} must contain at least one non-empty reference")
    return refs


# Friendly aliases for callers that prefer the role-oriented name.
NemotronAgentAdapters = AutonomousAgentAdapters
BoundedReasoningAgents = AutonomousAgentAdapters
ReasoningAgentAdapters = AutonomousAgentAdapters


__all__ = [
    "QLORA_SEARCH_SPACE",
    "AutonomousAgentAdapters",
    "BoundedReasoningAgents",
    "CuratedDatasetPlan",
    "DuplicateHypothesisError",
    "FailureCluster",
    "NemotronAgentAdapters",
    "ProviderHandoffError",
    "QLoRAConfig",
    "ReasoningAgentAdapters",
    "ResearchHypothesis",
    "hypothesis_fingerprint",
    "validate_qlora_config",
]
