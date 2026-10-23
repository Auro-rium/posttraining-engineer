"""The deterministic promotion gate. Nothing here calls an LLM; it fails closed.

Rules (every failed rule adds a reason; any reason means REJECT):

1. Both sides carry verified evidence and identical suite/split/seed/judged_by/served_by.
2. Both sides report an integer ``n`` > 0 (a missing ``n`` rejects: significance cannot be judged).
3. Parse rate: the candidate must report ``parse_rate`` and have ``parse_rate >=
   max(MIN_PARSE_RATE, base parse_rate - PARSE_TOLERANCE)``. A model that does not emit tool calls
   is invalid, not merely low-scoring (floor 0.90; never more than 0.02 below the base).
4. Relative improvement >= 10% and no absolute drop > 0.05 on the evaluated split.
5. Significance, one-sided at 5%: an exact paired sign (McNemar) test over per-task verdicts when
   both results carry ``verdicts`` for the same task set (``tasks_sha256``), else the unpaired
   two-proportion z-test.
6. Regression split (``champion_regression`` / ``candidate_regression``): same validity rules, and
   the candidate may not lose more than 0.05 absolute against the base there. Without both
   results the gate adds a warning (``regression_checked: false``); ``require_regression=True``
   turns that into a rejection.

The verdict reports the minimum detectable effect (80% power, one-sided 5%) so an underpowered
comparison is visible instead of being read as "no effect".
"""

from __future__ import annotations

import math
from math import isfinite
from typing import Any

VERIFIED: frozenset[str] = frozenset({"LIVE", "PRIOR_VERIFIED_RUN"})
MIN_RELATIVE_IMPROVEMENT = 0.10
MIN_Z = 1.645  # one-sided 5% two-proportion test: the gain must not be sampling noise
ALPHA = 0.05
MAX_ABSOLUTE_REGRESSION = 0.05
MIN_PARSE_RATE = 0.90
PARSE_TOLERANCE = 0.02
RECOMMENDED_N = 300
_Z_POWER80 = 0.8416


def _is_n(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _score(res: dict[str, Any]) -> float | None:
    try:
        v = float(res["score"])
    except (KeyError, TypeError, ValueError):
        return None
    return v if isfinite(v) else None


def sign_test_p(wins: int, losses: int) -> float:
    """Exact one-sided P(X >= wins) for X ~ Binomial(wins + losses, 0.5) (McNemar's exact test
    over the discordant pairs)."""
    m = wins + losses
    if m == 0:
        return 1.0
    return sum(math.comb(m, k) for k in range(wins, m + 1)) / 2 ** m


def paired_counts(champion: dict[str, Any], candidate: dict[str, Any]) -> tuple[int, int] | None:
    """``(candidate-only wins, champion-only wins)`` from per-task verdict strings, or None when
    they are absent, differ in length, or were not computed over the same task set."""
    a, b = champion.get("verdicts"), candidate.get("verdicts")
    if not (isinstance(a, str) and isinstance(b, str) and a and len(a) == len(b)
            and set(a + b) <= {"0", "1"}):
        return None
    sa, sb = champion.get("tasks_sha256"), candidate.get("tasks_sha256")
    if not (sa and sa == sb):
        return None
    wins = sum(x == "0" and y == "1" for x, y in zip(a, b, strict=True))
    losses = sum(x == "1" and y == "0" for x, y in zip(a, b, strict=True))
    return wins, losses


def improvement_z(champion: dict[str, Any], candidate: dict[str, Any],
                  c: float, n: float) -> float | None:
    """One-sided two-proportion z for candidate > champion; None if sizes are not usable."""
    n1, n2 = champion.get("n"), candidate.get("n")
    if not (_is_n(n1) and _is_n(n2)):
        return None
    if not (0.0 <= c <= 1.0 and 0.0 <= n <= 1.0):
        return None
    pooled = (c * n1 + n * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return math.inf if n > c else 0.0
    return (n - c) / se


def min_detectable_effect(p: float, n: int) -> float:
    """Smallest absolute gain detectable with 80% power at one-sided 5% (unpaired normal
    approximation, ``p`` the base accuracy, ``n`` tasks per side)."""
    p = min(max(p, 0.05), 0.95)
    return (MIN_Z + _Z_POWER80) * math.sqrt(2 * p * (1 - p) / n)


def parse_floor(base: dict[str, Any]) -> float:
    bpr = base.get("parse_rate")
    base_rate = float(bpr) if isinstance(bpr, int | float) and not isinstance(bpr, bool) else 0.0
    return max(MIN_PARSE_RATE, base_rate - PARSE_TOLERANCE)


def _parse_problem(side: str, base: dict[str, Any], res: dict[str, Any]) -> str | None:
    pr = res.get("parse_rate")
    if not (isinstance(pr, int | float) and not isinstance(pr, bool) and isfinite(pr)):
        return f"{side} parse_rate missing: cannot tell a wrong move from no tool call"
    floor = parse_floor(base)
    if pr < floor:
        return (f"{side} parse_rate {pr:.3f} < required {floor:.3f}: the model does not reliably "
                "emit tool calls (invalid, not merely low-scoring)")
    return None


def _regression_reasons(base: Any, cand: Any) -> list[str]:
    out: list[str] = []
    for side, res in (("champion", base), ("candidate", cand)):
        if not isinstance(res, dict) or _score(res) is None:
            out.append(f"{side} regression result missing or has no score")
    if out:
        return out
    for side, res in (("champion", base), ("candidate", cand)):
        if res.get("evidence") not in VERIFIED:
            out.append(f"{side} regression evidence {res.get('evidence')!r} is not verified")
        if not _is_n(res.get("n")):
            out.append(f"{side} regression n missing")
    for key in ("suite", "split", "seed", "judged_by", "served_by"):
        if base.get(key) != cand.get(key):
            out.append(f"regression {key} differs between champion and candidate")
    if cand.get("split") != "regression":
        out.append("regression results must come from the regression split")
    if problem := _parse_problem("candidate regression", base, cand):
        out.append(problem)
    drop = _score(base) - _score(cand)  # type: ignore[operator]
    if drop > MAX_ABSOLUTE_REGRESSION:
        out.append(f"regression split dropped {drop:.3f} > {MAX_ABSOLUTE_REGRESSION} "
                   f"({_score(base):.3f} -> {_score(cand):.3f})")
    return out


def promotion_gate(champion: dict[str, Any], candidate: dict[str, Any], *,
                   champion_regression: dict[str, Any] | None = None,
                   candidate_regression: dict[str, Any] | None = None,
                   require_regression: bool = False) -> dict[str, Any]:
    """Fail-closed promotion decision; see the module docstring for the rules."""
    reasons: list[str] = []
    warnings: list[str] = []
    c, n = _score(champion), _score(candidate)
    if c is None or n is None:
        return {"decision": "REJECT", "reasons": ["scores missing, non-numeric or not finite"]}
    for side, res in (("champion", champion), ("candidate", candidate)):
        if res.get("evidence") not in VERIFIED:
            reasons.append(f"{side} evidence {res.get('evidence')!r} is not verified")
    # judged_by / served_by: never compare two judges or two serving paths (sandbox CPU bf16
    # transformers vs serverless). Absent on both sides (older results) compares equal.
    for key in ("suite", "split", "seed", "judged_by", "served_by"):
        if champion.get(key) != candidate.get(key):
            reasons.append(f"{key} differs between champion and candidate")
    n1, n2 = champion.get("n"), candidate.get("n")
    sized = _is_n(n1) and _is_n(n2)
    if not sized:
        reasons.append("n missing or invalid on champion or candidate: significance cannot be "
                       "judged (fail-closed)")
    if problem := _parse_problem("candidate", champion, candidate):
        reasons.append(problem)
    delta = n - c
    relative = delta / abs(c) if c else None
    if (relative is None and n <= 0) or (relative is not None and relative < MIN_RELATIVE_IMPROVEMENT):
        reasons.append("minimum relative improvement not met")
    if max(0.0, -delta) > MAX_ABSOLUTE_REGRESSION:
        reasons.append("maximum regression exceeded")

    test, p_value, z = "none", None, None
    pairs = paired_counts(champion, candidate) if sized else None
    sha_a, sha_b = champion.get("tasks_sha256"), candidate.get("tasks_sha256")
    if sha_a and sha_b and sha_a != sha_b:
        reasons.append("tasks_sha256 differs: the two results were not computed on the same tasks")
    if pairs is not None:
        wins, losses = pairs
        test, p_value = "paired-exact (McNemar sign test)", sign_test_p(wins, losses)
        if p_value >= ALPHA:
            reasons.append(f"improvement not statistically significant (paired exact p="
                           f"{p_value:.4f} >= {ALPHA}; candidate-only correct {wins}, "
                           f"champion-only correct {losses})")
    elif sized:
        z = improvement_z(champion, candidate, c, n)
        if z is not None:
            test = "two-proportion z (unpaired; no per-task verdicts)"
            if z < MIN_Z:
                reasons.append(f"improvement not statistically significant (z={z:.2f} < {MIN_Z}, "
                               f"n={n1}/{n2})")
    out: dict[str, Any] = {"champion_score": c, "candidate_score": n,
                           "relative_improvement": relative, "test": test,
                           "z": None if z is None else round(z, 3),
                           "p_value": None if p_value is None else round(p_value, 5)}
    if sized:
        mde = min_detectable_effect(c, min(n1, n2))
        out["mde_80pct_power"] = round(mde, 4)
        out["power_note"] = (f"with n={min(n1, n2)} per side only a gain of about {mde:.3f} "
                             "absolute or more is detectable at 80% power (one-sided 5%, "
                             "unpaired approximation)")
        if 0 < delta < mde:
            warnings.append("observed gain is below the minimum detectable effect: the "
                            "comparison is underpowered")
        if min(n1, n2) < RECOMMENDED_N:
            warnings.append(f"n={min(n1, n2)} is below the recommended {RECOMMENDED_N} held-out tasks")

    have = (champion_regression is not None, candidate_regression is not None)
    if all(have):
        reg = _regression_reasons(champion_regression, candidate_regression)
        reasons += reg
        out["regression_checked"] = True
        out["regression"] = {"champion_score": _score(champion_regression),
                             "candidate_score": _score(candidate_regression),
                             "n": candidate_regression.get("n"), "ok": not reg}
    elif any(have):
        reasons.append("regression result supplied for only one side")
        out["regression_checked"] = False
    else:
        out["regression_checked"] = False
        (reasons if require_regression else warnings).append(
            "regression split not evaluated: forgetting was not checked")
    out.update({"decision": "REJECT" if reasons else "PROMOTE", "reasons": reasons,
                "warnings": warnings})
    return out
