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
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Protocol

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

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
_QLORA_SEARCH_SPACE: Final[Mapping[str, tuple[Any, ...]]] = MappingProxyType({
    "rank": (8, 16, 32),
    "alpha": (16, 32, 64),
    "dropout": (0.0, 0.05, 0.1),
    "learning_rate": (1e-4, 2e-4, 5e-4),
    "epochs": (1, 2, 3),
    "sequence_length": (512, 1024),
    "batch_size": (1, 2, 4),
    "gradient_accumulation_steps": (4, 8, 16),
    "target_modules": (("q_proj", "k_proj", "v_proj", "o_proj"),),
})
QLORA_SEARCH_SPACE: Final[Mapping[str, tuple[Any, ...]]] = _QLORA_SEARCH_SPACE
_TARGET_MODULES: Final[tuple[str, ...]] = ("q_proj", "k_proj", "v_proj", "o_proj")
_REFERENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?:traj|trajectory|artifact|dataset|eval|hypothesis|checkpoint|job|run|s3)://"
    r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,511}$"
)
_HISTORY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "run_id",
        "run_number",
        "experiment_id",
        "experiment_number",
        "status",
        "hypothesis_id",
        "fingerprint",
        "cluster_id",
        "statement",
        "prediction",
        "falsifier",
        "testable_prediction",
        "falsifiable_criterion",
        "dataset_id",
        "evidence_ids",
        "training_config",
        "evidence_refs",
        "artifact_ids",
        "provider_job_ids",
        "metrics",
        "evidence_class",
        "stop_reason",
        "created_at",
        "updated_at",
    }
)
_SUCCESS_STATUSES: Final[frozenset[str]] = frozenset({"SUCCEEDED", "OK"})
_VERIFIED_CLASSES: Final[frozenset[str]] = frozenset({"LIVE", "PRIOR_VERIFIED_RUN"})


class _HandoffModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FailureCluster(_HandoffModel):
    """One verified, train-side failure pattern."""

    cluster_id: StrictStr = Field(min_length=1)
    failure_type: StrictStr = Field(min_length=1)
    description: StrictStr = Field(min_length=1)
    count: StrictInt = Field(
        ge=1,
        validation_alias=AliasChoices("count", "example_count"),
    )
    evidence_refs: tuple[StrictStr, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "evidence_refs", "trajectory_refs", "evidence_references", "example_trajectories"
        ),
    )
    evidence_class: StrictStr = "LIVE"

    @model_validator(mode="after")
    def validate_evidence(self) -> FailureCluster:
        if self.evidence_class not in _EVIDENCE_LABELS:
            raise ValueError("evidence_class must be LIVE, PRIOR_VERIFIED_RUN, or EXPLANATION")
        if self.evidence_class not in _VERIFIED_EVIDENCE_LABELS:
            raise ValueError("failure clusters require verified evidence")
        if any(not _REFERENCE_PATTERN.fullmatch(ref) for ref in self.evidence_refs):
            raise ValueError("evidence_refs must contain non-empty references")
        return self

    @property
    def trajectory_refs(self) -> tuple[str, ...]:
        """Compatibility name used by trajectory-oriented callers."""

        return self.evidence_refs


class ResearchHypothesis(_HandoffModel):
    """A falsifiable repair hypothesis grounded in observed failures."""

    hypothesis_id: StrictStr = Field(min_length=1)
    cluster_id: StrictStr = Field(min_length=1)
    statement: StrictStr = Field(min_length=1)
    prediction: StrictStr = Field(
        min_length=1,
        validation_alias=AliasChoices("prediction", "testable_prediction"),
    )
    falsifier: StrictStr = Field(
        min_length=1,
        validation_alias=AliasChoices("falsifier", "falsifiable_criterion"),
    )
    evidence_refs: tuple[StrictStr, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices("evidence_refs", "evidence_references"),
    )
    confidence: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    evidence_class: StrictStr = "EXPLANATION"

    @field_validator("confidence", mode="before")
    @classmethod
    def require_json_float(cls, value: Any) -> Any:
        if value is not None and type(value) is not float:
            raise ValueError("confidence must be a JSON float")
        return value

    @model_validator(mode="after")
    def validate_evidence(self) -> ResearchHypothesis:
        if self.evidence_class not in _EVIDENCE_LABELS:
            raise ValueError("evidence_class must be LIVE, PRIOR_VERIFIED_RUN, or EXPLANATION")
        if not self.evidence_refs or any(
            not _REFERENCE_PATTERN.fullmatch(ref) for ref in self.evidence_refs
        ):
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

    plan_id: StrictStr = Field(
        min_length=1,
        validation_alias=AliasChoices("plan_id", "dataset_plan_id"),
    )
    selected_trajectory_refs: tuple[StrictStr, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "selected_trajectory_refs", "trajectory_references", "selected_record_ids"
        ),
    )
    dataset_artifact_ref: StrictStr = Field(
        min_length=1,
        validation_alias=AliasChoices("dataset_artifact_ref", "dataset_reference"),
    )
    record_count: StrictInt | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("record_count", "selected_record_count"),
    )
    evidence_class: StrictStr = "LIVE"

    @model_validator(mode="after")
    def validate_evidence(self) -> CuratedDatasetPlan:
        if self.evidence_class not in _VERIFIED_EVIDENCE_LABELS:
            raise ValueError("curated datasets require LIVE or PRIOR_VERIFIED_RUN evidence")
        if any(not _REFERENCE_PATTERN.fullmatch(ref) for ref in self.selected_trajectory_refs):
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

    rank: StrictInt
    alpha: StrictInt
    dropout: StrictFloat
    learning_rate: StrictFloat
    epochs: StrictInt
    sequence_length: StrictInt
    batch_size: StrictInt
    gradient_accumulation_steps: StrictInt
    target_modules: tuple[StrictStr, ...]

    @field_validator("dropout", "learning_rate", mode="before")
    @classmethod
    def require_json_float(cls, value: Any) -> Any:
        if type(value) is not float:
            raise ValueError("QLoRA float fields must be JSON floats")
        return value

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
        raw: Mapping[str, Any] | None = dict(item) if isinstance(item, Mapping) else None
        if raw is None:
            dump = getattr(item, "model_dump", None)
            if callable(dump):
                dumped = dump(mode="json")
                if isinstance(dumped, Mapping):
                    raw = dumped
        if raw is not None:
            unknown = set(raw).difference(_HISTORY_KEYS)
            if unknown:
                raise ValueError(
                    f"experiment history contains non-metadata keys: {sorted(unknown)}"
                )
            selected = {
                key: raw[key]
                for key in _HISTORY_KEYS.difference({"created_at", "updated_at"})
                if key in raw
            }
            _validate_history_metadata(selected)
            result.append(selected)
            continue
        raise TypeError("experiment_history entries must be mappings or Pydantic records")
    return result


def _validate_history_metadata(item: Mapping[str, Any]) -> None:
    """Validate the small, metadata-only subset allowed into Nemotron context."""

    for key, value in item.items():
        if key in {
            "evidence_refs",
            "evidence_ids",
            "artifact_ids",
            "provider_job_ids",
            "dataset_id",
        }:
            if isinstance(value, str):
                refs = [value]
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                refs = list(value)
            else:
                raise TypeError(f"history {key} must contain opaque references")
            _references(refs, f"history.{key}")
        elif key == "metrics":
            if not isinstance(value, Mapping):
                raise TypeError("history.metrics must be a mapping")
            for metric_name, metric_value in value.items():
                if (
                    not isinstance(metric_name, str)
                    or not metric_name
                    or isinstance(metric_value, bool)
                    or not isinstance(metric_value, (int, float))
                    or not math.isfinite(float(metric_value))
                ):
                    raise ValueError("history.metrics must contain finite numeric values")
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"history.{key} is not metadata")


def _collection(response: Any, key: str) -> tuple[list[Any], str]:
    decoded = _json_response(response)
    if not isinstance(decoded, Mapping):
        raise ProviderHandoffError("provider response must be a JSON object or list")
    required = {"status", "evidence_class", key}
    if set(decoded) != required:
        unknown = set(decoded).difference(required)
        if unknown:
            raise ProviderHandoffError(
                f"provider response contains unknown keys: {sorted(unknown)}"
            )
        raise ProviderHandoffError(
            f"provider response must contain exactly {sorted(required)}"
        )
    status = decoded["status"]
    evidence_class = decoded["evidence_class"]
    if status not in _SUCCESS_STATUSES or not isinstance(evidence_class, str):
        raise ProviderHandoffError("provider response has invalid status or evidence_class")
    if evidence_class not in _EVIDENCE_LABELS:
        raise ProviderHandoffError("provider response has unknown evidence_class")
    values = decoded[key]
    if not isinstance(values, list):
        raise ProviderHandoffError(f"provider response field {key} must be a JSON list")
    return values, evidence_class


def _object_response(response: Any, key: str) -> tuple[dict[str, Any], str]:
    decoded = _json_response(response)
    if not isinstance(decoded, Mapping):
        raise ProviderHandoffError("provider response must be a JSON object")
    required = {"status", "evidence_class", key}
    if set(decoded) != required:
        unknown = set(decoded).difference(required)
        if unknown:
            raise ProviderHandoffError(
                f"provider response contains unknown keys: {sorted(unknown)}"
            )
        raise ProviderHandoffError(
            f"provider response must contain exactly {sorted(required)}"
        )
    status = decoded["status"]
    evidence_class = decoded["evidence_class"]
    if status not in _SUCCESS_STATUSES or not isinstance(evidence_class, str):
        raise ProviderHandoffError("provider response has invalid status or evidence_class")
    if evidence_class not in _EVIDENCE_LABELS:
        raise ProviderHandoffError("provider response has unknown evidence_class")
    value = decoded[key]
    if not isinstance(value, Mapping):
        raise ProviderHandoffError(f"provider response field {key} must be a JSON object")
    return dict(value), evidence_class


@dataclass(frozen=True)
class AutonomousAgentAdapters:
    """Four bounded Nemotron handoffs used by the autonomous supervisor."""

    provider: NemotronProvider

    def __post_init__(self) -> None:
        provider_model_id = getattr(self.provider, "model_id", None)
        if provider_model_id != NEMOTRON_MODEL_ID:
            raise ValueError(
                "autonomous reasoning requires pinned Nemotron model "
                f"{NEMOTRON_MODEL_ID}; received {provider_model_id!r}"
            )
        if not callable(getattr(self.provider, "invoke", None)):
            raise TypeError("provider must expose invoke(prompt, *, agent_name, system_prompt)")

    def _call(self, agent_key: str, payload: Mapping[str, Any]) -> Any:
        contract = get_prompt_contract(agent_key)
        prompt = contract.render_handoff(payload)
        target = self.provider.invoke
        try:
            return target(
                prompt,
                agent_name=agent_key,
                system_prompt=contract.prompt,
            )
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
            raw_items, response_class = _collection(response, "clusters")
            if response_class not in _VERIFIED_CLASSES:
                raise ProviderHandoffError("failure analysis must cite verified evidence")
            for item in raw_items:
                if not isinstance(item, Mapping) or item.get("evidence_class") != response_class:
                    raise ProviderHandoffError(
                        "cluster evidence_class must match response evidence_class"
                    )
            clusters = tuple(FailureCluster.model_validate(item) for item in raw_items)
            if any(not set(item.evidence_refs).issubset(set(refs)) for item in clusters):
                raise ProviderHandoffError(
                    "cluster evidence_refs are not a subset of verified references"
                )
            return clusters
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
        verified_evidence_references: Sequence[str] | None = None,
        verified_evidence_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[ResearchHypothesis, ...]:
        cluster_models = tuple(
            item if isinstance(item, FailureCluster) else FailureCluster.model_validate(item)
            for item in failure_clusters
        )
        cluster_refs = {ref for item in cluster_models for ref in item.evidence_refs}
        if verified_evidence_references is None:
            raise ProviderHandoffError("research requires coordinator-verified evidence references")
        verified_refs = set(
            _references(verified_evidence_references, "verified_evidence_references")
        )
        if not cluster_refs.issubset(verified_refs):
            raise ProviderHandoffError("failure-cluster evidence is not coordinator-verified")
        evidence_metadata = _coordinator_evidence_metadata(
            verified_evidence_metadata, verified_refs
        )
        clusters = [item.model_dump(mode="json") for item in cluster_models]
        history = _mapping_history(experiment_history)
        response = self._call(
            "ResearchAgent",
            {
                "failure_clusters": clusters,
                "verified_evidence_references": sorted(verified_refs),
                "verified_evidence_metadata": evidence_metadata,
                "experiment_history": history,
            },
        )
        try:
            raw_items, response_class = _collection(response, "hypotheses")
            if response_class != "EXPLANATION":
                raise ProviderHandoffError("research hypotheses must be EXPLANATION evidence")
            for item in raw_items:
                if not isinstance(item, Mapping) or item.get("evidence_class") != response_class:
                    raise ProviderHandoffError(
                        "hypothesis evidence_class must match response evidence_class"
                    )
            hypotheses = tuple(
                ResearchHypothesis.model_validate(item) for item in raw_items
            )
            if any(not set(item.evidence_refs).issubset(verified_refs) for item in hypotheses):
                raise ProviderHandoffError("hypothesis evidence_refs are not coordinator-verified")
        except ProviderHandoffError:
            raise
        except Exception as exc:
            raise ProviderHandoffError("provider research JSON failed schema validation") from exc
        self._reject_failed_duplicates(
            hypotheses,
            history,
            verified_refs=verified_refs,
            evidence_metadata=evidence_metadata,
        )
        return hypotheses

    research_hypotheses = research

    def curate(
        self,
        verified_trajectory_references: Sequence[str],
        hypotheses: Sequence[ResearchHypothesis | Mapping[str, Any]] = (),
        experiment_history: Sequence[Any] = (),
        verified_dataset_artifact_references: Sequence[str] | None = None,
    ) -> CuratedDatasetPlan:
        refs = _references(verified_trajectory_references, "verified_trajectory_references")
        history = _mapping_history(experiment_history)
        hypothesis_models = tuple(
            item
            if isinstance(item, ResearchHypothesis)
            else ResearchHypothesis.model_validate(item)
            for item in hypotheses
        )
        verified_refs = set(refs)
        if any(not set(item.evidence_refs).issubset(verified_refs) for item in hypothesis_models):
            raise ProviderHandoffError("hypothesis evidence_refs are not coordinator-verified")
        if verified_dataset_artifact_references is None:
            raise ProviderHandoffError(
                "curation requires coordinator-owned dataset artifact provenance"
            )
        verified_dataset_refs = set(
            _references(
                verified_dataset_artifact_references,
                "verified_dataset_artifact_references",
            )
        )
        hypothesis_values = [item.model_dump(mode="json") for item in hypothesis_models]
        response = self._call(
            "DataCuratorAgent",
            {
                "verified_trajectory_references": refs,
                "hypotheses": hypothesis_values,
                "experiment_history": history,
            },
        )
        try:
            raw_plan, response_class = _object_response(response, "plan")
            if response_class not in _VERIFIED_CLASSES:
                raise ProviderHandoffError("curation must cite verified evidence")
            if raw_plan.get("evidence_class") != response_class:
                raise ProviderHandoffError("plan evidence_class must match response evidence_class")
            plan = CuratedDatasetPlan.model_validate(raw_plan)
        except ProviderHandoffError:
            raise
        except Exception as exc:
            raise ProviderHandoffError("provider curation JSON failed schema validation") from exc
        if not set(plan.selected_trajectory_refs).issubset(set(refs)):
            raise ProviderHandoffError(
                "curation selected a trajectory outside verified input references"
            )
        if plan.dataset_artifact_ref not in verified_dataset_refs:
            raise ProviderHandoffError(
                "curation returned a dataset artifact without coordinator provenance"
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
            raw_config, response_class = _object_response(response, "config")
            if response_class != "EXPLANATION":
                raise ProviderHandoffError("QLoRA design must be EXPLANATION evidence")
            return validate_qlora_config(raw_config)
        except ProviderHandoffError:
            raise
        except Exception as exc:
            raise ProviderHandoffError(
                "provider QLoRA JSON failed bounded schema validation"
            ) from exc

    design_training = design_qlora

    @staticmethod
    def _reject_failed_duplicates(
        hypotheses: Sequence[ResearchHypothesis],
        history: Sequence[Mapping[str, Any]],
        *,
        verified_refs: set[str],
        evidence_metadata: Mapping[str, Mapping[str, Any]],
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
            new_refs = current_refs.difference(previous_refs)
            if new_refs and not new_refs.issubset(verified_refs):
                raise DuplicateHypothesisError(
                    f"hypothesis {hypothesis.hypothesis_id!r} cites unverified new evidence"
                )
            if new_refs and not new_refs.issubset(evidence_metadata):
                raise DuplicateHypothesisError(
                    f"hypothesis {hypothesis.hypothesis_id!r} cites evidence without "
                    "a new measurement"
                )
            # A new reference is accepted only when it is coordinator-owned
            # and has not merely been appended to bypass this guard.  The
            # current run's verified references are the coordinator's source
            # of truth; without one, the proposal remains a duplicate.
            if not new_refs:
                raise DuplicateHypothesisError(
                    f"hypothesis {hypothesis.hypothesis_id!r} repeats a failed hypothesis"
                )


def _history_refs(item: Mapping[str, Any]) -> set[str]:
    for key in ("evidence_refs", "evidence_references", "trajectory_refs", "evidence_refs_used"):
        value = item.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return {str(ref) for ref in value}
    return set()


def _coordinator_evidence_metadata(
    metadata: Mapping[str, Mapping[str, Any]] | None,
    verified_refs: set[str],
) -> dict[str, dict[str, Any]]:
    """Validate coordinator-owned proof for evidence that bypasses a duplicate."""

    if metadata is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    allowed = {
        "verified",
        "run_id",
        "experiment_id",
        "measurement_id",
        "artifact_id",
        "evidence_class",
    }
    for ref, value in metadata.items():
        if not isinstance(ref, str) or not _REFERENCE_PATTERN.fullmatch(ref):
            raise ValueError("coordinator evidence metadata has an invalid reference")
        if ref not in verified_refs:
            raise ValueError("coordinator evidence metadata contains an unverified reference")
        if not isinstance(value, Mapping) or set(value).difference(allowed):
            raise ValueError("coordinator evidence metadata must be allowlisted metadata")
        if value.get("verified") is not True:
            raise ValueError("coordinator evidence must be explicitly verified")
        if not isinstance(value.get("run_id"), str) or not value["run_id"].strip():
            raise ValueError("coordinator evidence requires an owning run_id")
        measurement = value.get("measurement_id", value.get("artifact_id"))
        if not isinstance(measurement, str) or not _REFERENCE_PATTERN.fullmatch(measurement):
            raise ValueError("coordinator evidence requires a verified measurement or artifact")
        if value.get("evidence_class") not in _VERIFIED_CLASSES:
            raise ValueError("coordinator evidence must use verified evidence_class")
        result[ref] = dict(value)
    return result


def _references(value: Sequence[str], name: str) -> list[str]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of opaque references")
    refs = list(value)
    if not refs or any(
        not isinstance(ref, str) or not _REFERENCE_PATTERN.fullmatch(ref) for ref in refs
    ):
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
