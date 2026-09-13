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
from app.objective.models import CorrectionProposal, decode_trajectory_reference


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
    """Judgment-only selection and replay proposals; the worker owns admission."""

    plan_id: StrictStr = Field(
        min_length=1,
        validation_alias=AliasChoices("plan_id", "dataset_plan_id"),
    )
    selected_trajectory_refs: tuple[StrictStr, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices(
            "selected_trajectory_refs", "trajectory_references", "selected_record_ids"
        ),
    )
    correction_proposals: tuple[CorrectionProposal, ...] = Field(
        default_factory=tuple, max_length=100
    )
    target_failure_classes: tuple[StrictStr, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices("target_failure_classes", "failure_classes"),
    )
    record_count: StrictInt = Field(
        ge=0,
        validation_alias=AliasChoices("record_count", "selected_record_count"),
    )
    evidence_class: StrictStr

    @model_validator(mode="after")
    def validate_evidence(self) -> CuratedDatasetPlan:
        if self.evidence_class not in _VERIFIED_EVIDENCE_LABELS:
            raise ValueError("curated datasets require LIVE or PRIOR_VERIFIED_RUN evidence")
        if not self.selected_trajectory_refs and not self.correction_proposals:
            raise ValueError("curation plan requires selected trajectories or correction proposals")
        if any(not _REFERENCE_PATTERN.fullmatch(ref) for ref in self.selected_trajectory_refs):
            raise ValueError("selected_trajectory_refs must contain non-empty references")
        if self.record_count != len(self.selected_trajectory_refs):
            raise ValueError("record_count must equal selected_trajectory_refs length")
        if len(set(self.selected_trajectory_refs)) != len(self.selected_trajectory_refs):
            raise ValueError("selected_trajectory_refs must be unique")
        if len({item.proposal_id for item in self.correction_proposals}) != len(
            self.correction_proposals
        ):
            raise ValueError("correction proposals must be unique")
        if len(set(self.target_failure_classes)) != len(self.target_failure_classes) or any(
            not item.strip() for item in self.target_failure_classes
        ):
            raise ValueError("target_failure_classes must be non-empty and unique")
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
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProviderHandoffError("provider returned invalid UTF-8 JSON") from exc
    if not isinstance(raw, str):
        message = getattr(raw, "message", None)
        if message is not None:
            if not isinstance(message, Mapping):
                raise ProviderHandoffError("Strands AgentResult message must be an object")
            blocks = message.get("content")
            if not isinstance(blocks, list) or not blocks:
                raise ProviderHandoffError("Strands AgentResult has no text content blocks")
            text_blocks: list[str] = []
            for block in blocks:
                if (
                    not isinstance(block, Mapping)
                    or set(block) != {"text"}
                    or not isinstance(block.get("text"), str)
                ):
                    raise ProviderHandoffError(
                        "Strands AgentResult contains a non-text or malformed content block"
                    )
                text_blocks.append(block["text"])
            # Strands represents the assistant turn as text content blocks;
            # concatenate them without coercing other block types to strings.
            raw = "".join(text_blocks)
        else:
            for attribute in ("text", "output", "content"):
                value = getattr(raw, attribute, None)
                if value is not None:
                    return _json_response(value)
            raise ProviderHandoffError("provider returned a non-JSON response")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProviderHandoffError(f"provider JSON contains duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_non_json_constant(value: str) -> None:
        raise ProviderHandoffError(f"provider JSON contains invalid constant: {value}")

    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_json_constant,
        )
    except ProviderHandoffError:
        raise
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
                if key in raw and not (key == "dataset_id" and raw[key] is None)
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
            if not refs and key != "dataset_id":
                continue
            if key == "dataset_id" and len(refs) == 1 and not _REFERENCE_PATTERN.fullmatch(refs[0]):
                if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", refs[0]) is None:
                    raise ValueError("history.dataset_id must be an opaque dataset ID")
            else:
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
        elif key == "training_config":
            if not isinstance(value, Mapping):
                raise TypeError("history.training_config must be a metadata mapping")
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
        *,
        evidence_class: str,
    ) -> tuple[FailureCluster, ...]:
        refs = _references(trajectory_references, "trajectory_references")
        if evidence_class not in _VERIFIED_CLASSES:
            raise ProviderHandoffError(
                "failure analysis requires coordinator-verified evidence_class"
            )
        history = _mapping_history(experiment_history)
        response = self._call(
            "FailureAnalystAgent",
            {
                "trajectory_references": refs,
                "experiment_history": history,
                "evidence_class": evidence_class,
            },
        )
        try:
            raw_items, response_class = _collection(response, "clusters")
            if response_class != evidence_class:
                raise ProviderHandoffError(
                    "failure-analysis evidence_class must match coordinator provenance"
                )
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
        *,
        run_id: str,
        experiment_number: int,
        verified_evidence_references: Sequence[str] | None = None,
        verified_evidence_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[ResearchHypothesis, ...]:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ProviderHandoffError("research requires coordinator run_id")
        if type(experiment_number) is not int or not 1 <= experiment_number <= 5:
            raise ProviderHandoffError("research requires experiment_number from 1 through 5")
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
            verified_evidence_metadata,
            verified_refs,
            run_id=run_id,
            experiment_number=experiment_number,
        )
        clusters = [item.model_dump(mode="json") for item in cluster_models]
        history = _mapping_history(experiment_history)
        response = self._call(
            "ResearchAgent",
            {
                "failure_clusters": clusters,
                "run_id": run_id,
                "experiment_number": experiment_number,
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
            run_id=run_id,
            experiment_number=experiment_number,
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
        *,
        failure_clusters: Sequence[FailureCluster | Mapping[str, Any]],
        verified_trajectory_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> CuratedDatasetPlan:
        refs = _references(verified_trajectory_references, "verified_trajectory_references")
        history = _mapping_history(experiment_history)
        cluster_models = tuple(
            item if isinstance(item, FailureCluster) else FailureCluster.model_validate(item)
            for item in failure_clusters
        )
        if not cluster_models:
            raise ProviderHandoffError("curation requires coordinator-verified failure clusters")
        if any(not set(item.evidence_refs).issubset(set(refs)) for item in cluster_models):
            raise ProviderHandoffError("failure-cluster evidence is not coordinator-verified")
        hypothesis_models = tuple(
            item
            if isinstance(item, ResearchHypothesis)
            else ResearchHypothesis.model_validate(item)
            for item in hypotheses
        )
        verified_refs = set(refs)
        if any(not set(item.evidence_refs).issubset(verified_refs) for item in hypothesis_models):
            raise ProviderHandoffError("hypothesis evidence_refs are not coordinator-verified")
        cluster_ids = {item.cluster_id for item in cluster_models}
        if any(item.cluster_id not in cluster_ids for item in hypothesis_models):
            raise ProviderHandoffError(
                "hypothesis references a failure cluster outside coordinator input"
            )
        trajectory_metadata = _coordinator_trajectory_metadata(
            verified_trajectory_metadata, set(refs)
        )
        clusters = [item.model_dump(mode="json") for item in cluster_models]
        hypothesis_values = [item.model_dump(mode="json") for item in hypothesis_models]
        response = self._call(
            "DataCuratorAgent",
            {
                "verified_trajectory_references": refs,
                "verified_trajectory_metadata": trajectory_metadata,
                "failure_clusters": clusters,
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
        correction_refs: set[str] = set()
        failure_refs = {ref for cluster in cluster_models for ref in cluster.evidence_refs}
        for proposal in plan.correction_proposals:
            matching_reference = None
            for reference_value in refs:
                try:
                    reference = decode_trajectory_reference(reference_value)
                except ValueError:
                    continue
                if (
                    reference.trajectory_id == proposal.source_trajectory_id
                    and reference.task_id == proposal.task_id
                    and reference.split is proposal.split
                ):
                    matching_reference = reference_value
                    break
            if matching_reference is None:
                raise ProviderHandoffError(
                    "correction proposal source is outside verified input references"
                )
            if matching_reference not in failure_refs:
                raise ProviderHandoffError(
                    "correction proposal source is not coordinator-verified failure evidence"
                )
            correction_refs.add(matching_reference)
        allowed_failure_classes = {item.failure_type for item in cluster_models}
        if not set(plan.target_failure_classes).issubset(allowed_failure_classes):
            raise ProviderHandoffError(
                "curation selected a failure class outside coordinator-verified clusters"
            )
        plan_source_refs = set(plan.selected_trajectory_refs) | correction_refs
        selected_classes = {trajectory_metadata[ref]["evidence_class"] for ref in plan_source_refs}
        if len(selected_classes) != 1 or plan.evidence_class not in selected_classes:
            raise ProviderHandoffError(
                "curation evidence_class must match coordinator provenance for selected "
                "trajectories"
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
        # Action proposals belong only to the objective replay boundary.  The
        # training designer needs plan metadata, never candidate trajectories.
        plan.pop("correction_proposals", None)
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
        run_id: str,
        experiment_number: int,
        verified_refs: set[str],
        evidence_metadata: Mapping[str, Mapping[str, Any]],
    ) -> None:
        observed_fingerprints: set[str] = set()
        for hypothesis in hypotheses:
            if hypothesis.fingerprint in observed_fingerprints:
                raise DuplicateHypothesisError(
                    "provider returned duplicate hypotheses in the same response"
                )
            observed_fingerprints.add(hypothesis.fingerprint)

        failed = [
            item
            for item in history
            if str(item.get("status", "")).lower() in {"failed", "rejected"}
            and item.get("run_id", run_id) == run_id
        ]
        prior: dict[str, list[tuple[set[str], int | None]]] = {}
        for item in failed:
            fingerprint = item.get("fingerprint")
            if isinstance(fingerprint, str):
                prior.setdefault(fingerprint, []).append(
                    (_history_refs(item), _history_experiment_number(item))
                )
            try:
                prior.setdefault(hypothesis_fingerprint(item), []).append(
                    (_history_refs(item), _history_experiment_number(item))
                )
            except (TypeError, ValueError):
                continue
        for hypothesis in hypotheses:
            previous_attempts = prior.get(hypothesis.fingerprint)
            if not previous_attempts:
                continue
            current_refs = set(hypothesis.evidence_refs)
            for previous_refs, previous_experiment_number in previous_attempts:
                if previous_experiment_number is None:
                    raise DuplicateHypothesisError(
                        "failed hypothesis history is missing experiment_number scope"
                    )
                if experiment_number <= previous_experiment_number:
                    raise DuplicateHypothesisError(
                        f"hypothesis {hypothesis.hypothesis_id!r} repeats a failed hypothesis "
                        "within the same run/experiment scope"
                    )
                new_refs = current_refs.difference(previous_refs)
                if not new_refs:
                    raise DuplicateHypothesisError(
                        f"hypothesis {hypothesis.hypothesis_id!r} repeats a failed hypothesis"
                    )
                if not new_refs.issubset(verified_refs):
                    raise DuplicateHypothesisError(
                        f"hypothesis {hypothesis.hypothesis_id!r} cites unverified new evidence"
                    )
                if any(
                    evidence_metadata[ref]["run_id"] != run_id
                    or evidence_metadata[ref]["experiment_number"] != experiment_number
                    for ref in new_refs
                ):
                    raise DuplicateHypothesisError(
                        f"hypothesis {hypothesis.hypothesis_id!r} lacks new evidence for the "
                        "current run/experiment"
                    )


def _history_refs(item: Mapping[str, Any]) -> set[str]:
    references: set[str] = set()
    for key in (
        "evidence_refs",
        "evidence_references",
        "trajectory_refs",
        "evidence_refs_used",
        "evidence_ids",
    ):
        value = item.get(key)
        if isinstance(value, str):
            references.add(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            references.update(str(ref) for ref in value)
    return references


def _history_experiment_number(item: Mapping[str, Any]) -> int | None:
    value = item.get("experiment_number")
    return value if type(value) is int and 1 <= value <= 5 else None


def _coordinator_trajectory_metadata(
    metadata: Mapping[str, Mapping[str, Any]] | None,
    verified_refs: set[str],
) -> dict[str, dict[str, Any]]:
    """Validate coordinator-owned source provenance for every trajectory ref."""

    if metadata is None:
        raise ProviderHandoffError("handoff requires coordinator-verified trajectory metadata")
    if set(metadata) != verified_refs:
        raise ProviderHandoffError(
            "trajectory metadata must cover exactly the coordinator-verified references"
        )
    allowed = {
        "verified",
        "run_id",
        "experiment_number",
        "measurement_id",
        "artifact_id",
        "evidence_class",
    }
    result: dict[str, dict[str, Any]] = {}
    for ref, value in metadata.items():
        if not isinstance(value, Mapping) or set(value).difference(allowed):
            raise ProviderHandoffError(
                "trajectory metadata must be allowlisted coordinator metadata"
            )
        if value.get("verified") is not True:
            raise ProviderHandoffError("trajectory evidence must be explicitly verified")
        if (
            not isinstance(value.get("run_id"), str)
            or not value["run_id"].strip()
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", value["run_id"]) is None
        ):
            raise ProviderHandoffError("trajectory evidence requires an opaque source run_id")
        experiment_number = value.get("experiment_number")
        if type(experiment_number) is not int or not 1 <= experiment_number <= 5:
            raise ProviderHandoffError("trajectory evidence requires source experiment_number 1..5")
        if value.get("evidence_class") not in _VERIFIED_CLASSES:
            raise ProviderHandoffError("trajectory evidence must use LIVE or PRIOR_VERIFIED_RUN")
        artifact_ref = value.get("artifact_id", value.get("measurement_id"))
        if not isinstance(artifact_ref, str) or not _REFERENCE_PATTERN.fullmatch(artifact_ref):
            raise ProviderHandoffError("trajectory evidence requires a verified artifact reference")
        for key in ("artifact_id", "measurement_id"):
            candidate = value.get(key)
            if candidate is not None and (
                not isinstance(candidate, str) or not _REFERENCE_PATTERN.fullmatch(candidate)
            ):
                raise ProviderHandoffError(f"trajectory metadata {key} must be an opaque reference")
        result[ref] = dict(value)
    return result


def _coordinator_evidence_metadata(
    metadata: Mapping[str, Mapping[str, Any]] | None,
    verified_refs: set[str],
    *,
    run_id: str,
    experiment_number: int,
) -> dict[str, dict[str, Any]]:
    """Validate coordinator-owned proof and bind it to this run/experiment."""

    result = _coordinator_trajectory_metadata(metadata, verified_refs)
    for _, value in result.items():
        if value.get("run_id") != run_id:
            raise ValueError("coordinator evidence run_id must match the active run")
        if value.get("experiment_number") != experiment_number or type(
            value.get("experiment_number")
        ) is not int:
            raise ValueError(
                "coordinator evidence experiment_number must match the active experiment"
            )
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
    "CorrectionProposal",
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
