"""Small, deterministic non-parametric statistics utilities."""
from __future__ import annotations

import math
import random
import statistics


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Return Cliff's delta: ``(#a>b - #a<b) / (len(a) * len(b))``."""
    _require_non_empty(a, "a")
    _require_non_empty(b, "b")
    greater = 0
    less = 0
    for left in a:
        for right in b:
            greater += left > right
            less += left < right
    return (greater - less) / (len(a) * len(b))


def bootstrap_ci_median(
    xs: list[float],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a percentile-bootstrap confidence interval for the median."""
    _require_non_empty(xs, "xs")
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    _validate_alpha(alpha)

    rng = random.Random(seed)
    sample_size = len(xs)
    medians = sorted(
        statistics.median(rng.choices(xs, k=sample_size))
        for _ in range(n_boot)
    )
    return (
        _percentile(medians, alpha / 2),
        _percentile(medians, 1 - alpha / 2),
    )


def sign_test_pvalue(xs: list[float], mu: float = 0.0) -> float:
    """Return the exact two-sided sign-test p-value, excluding ties."""
    _require_non_empty(xs, "xs")
    positives = sum(value > mu for value in xs)
    negatives = sum(value < mu for value in xs)
    trials = positives + negatives
    if trials == 0:
        return 1.0

    tail_count = min(positives, negatives)
    tail_probability = sum(
        math.comb(trials, successes) for successes in range(tail_count + 1)
    ) / (2**trials)
    return min(1.0, 2 * tail_probability)


def holm_bonferroni(
    pvalues: list[float],
    alpha: float = 0.05,
) -> list[bool]:
    """Return Holm step-down rejection decisions in original input order."""
    _validate_pvalues(pvalues)
    _validate_alpha(alpha)

    reject = [False] * len(pvalues)
    ordered = sorted(enumerate(pvalues), key=lambda item: (item[1], item[0]))
    for rank, (original_index, pvalue) in enumerate(ordered):
        if pvalue > alpha / (len(pvalues) - rank):
            break
        reject[original_index] = True
    return reject


def benjamini_hochberg(
    pvalues: list[float],
    alpha: float = 0.05,
) -> list[bool]:
    """Return Benjamini-Hochberg FDR decisions in original input order."""
    _validate_pvalues(pvalues)
    _validate_alpha(alpha)

    ordered = sorted(enumerate(pvalues), key=lambda item: (item[1], item[0]))
    cutoff_rank = 0
    total = len(pvalues)
    for rank, (_, pvalue) in enumerate(ordered, start=1):
        if pvalue <= alpha * rank / total:
            cutoff_rank = rank

    reject = [False] * total
    for original_index, _ in ordered[:cutoff_rank]:
        reject[original_index] = True
    return reject


def _percentile(sorted_values: list[float], quantile: float) -> float:
    position = quantile * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    weight = position - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return float(lower + weight * (upper - lower))


def _require_non_empty(values: list[float], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} values must be finite")


def _validate_pvalues(pvalues: list[float]) -> None:
    if not pvalues:
        raise ValueError("pvalues must not be empty")
    if any(
        not math.isfinite(pvalue) or not 0 <= pvalue <= 1
        for pvalue in pvalues
    ):
        raise ValueError("pvalues must be finite and within [0, 1]")


def _validate_alpha(alpha: float) -> None:
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
