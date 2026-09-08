"""Fixed-seed benchmark and evaluation helpers.

These helpers are deliberately model-agnostic.  A caller provides a pure
predictor callback; this module fixes case ordering and computes metrics without
sampling, timestamps, or hidden state.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkCase[InputT, ExpectedT](BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    input: InputT
    expected: ExpectedT


class CaseEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    expected: Any
    predicted: Any
    passed: bool


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    total_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    case_order: tuple[str, ...]
    cases: tuple[CaseEvaluation, ...]


def seeded_case_order(
    cases: Sequence[BenchmarkCase[Any, Any]], seed: int
) -> tuple[BenchmarkCase[Any, Any], ...]:
    """Return a deterministic shuffled copy, leaving caller-owned input untouched."""

    ordered = list(cases)
    random.Random(seed).shuffle(ordered)
    return tuple(ordered)


def run_benchmark(
    cases: Iterable[BenchmarkCase[Any, Any]],
    predictor: Callable[[Any], Any],
    *,
    seed: int = 0,
) -> BenchmarkResult:
    """Run a predictor in deterministic seeded order and calculate accuracy."""

    ordered = seeded_case_order(tuple(cases), seed)
    evaluations = tuple(
        CaseEvaluation(
            case_id=case.case_id,
            expected=case.expected,
            predicted=(predicted := predictor(case.input)),
            passed=predicted == case.expected,
        )
        for case in ordered
    )
    passed = sum(item.passed for item in evaluations)
    total = len(evaluations)
    return BenchmarkResult(
        seed=seed,
        total_cases=total,
        passed_cases=passed,
        accuracy=passed / total if total else 0.0,
        case_order=tuple(case.case_id for case in ordered),
        cases=evaluations,
    )


def evaluate_predictions(
    expected: Sequence[Any], predicted: Sequence[Any], *, seed: int = 0
) -> BenchmarkResult:
    """Evaluate paired predictions through the same deterministic result schema."""

    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have equal lengths")
    cases: tuple[BenchmarkCase[Any, Any], ...] = tuple(
        BenchmarkCase(case_id=f"case-{index}", input=value, expected=target)
        for index, (value, target) in enumerate(zip(predicted, expected, strict=True))
    )
    return run_benchmark(cases, lambda value: value, seed=seed)


class FixedSeedBenchmark:
    """Reusable benchmark runner with a seed fixed at construction time."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def run(
        self,
        cases: Iterable[BenchmarkCase[Any, Any]],
        predictor: Callable[[Any], Any],
    ) -> BenchmarkResult:
        return run_benchmark(cases, predictor, seed=self.seed)


DeterministicBenchmark = FixedSeedBenchmark
FixedSeedEvaluator = FixedSeedBenchmark
evaluate = run_benchmark
run_evaluation = run_benchmark
seeded_evaluate = run_benchmark
