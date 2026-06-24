"""Permutation null-model significance for per-sample alignment."""
from __future__ import annotations

import random
import statistics

from scipy.stats import spearmanr


def _aligned_vals(attr: dict, causal: dict) -> tuple[list[float], list[float]]:
    """Return attribution and causal values aligned by integer-sorted layers."""
    if not attr or not causal:
        raise ValueError("attr and causal must not be empty")
    if set(attr) != set(causal):
        raise ValueError("attr and causal must share the same layers")

    try:
        layers = sorted(attr, key=lambda key: int(key))
    except (TypeError, ValueError) as exc:
        raise ValueError("layer keys must be integer-like strings") from exc

    return (
        [_to_float(attr[layer]) for layer in layers],
        [_to_float(causal[layer]) for layer in layers],
    )


def _spearman(a: list[float], c: list[float]) -> float:
    """Return Spearman rho, falling back to 0.0 for constant inputs."""
    if len(set(a)) < 2 or len(set(c)) < 2:
        return 0.0
    correlation = spearmanr(a, c).correlation
    if correlation is None or correlation != correlation:
        return 0.0
    return float(correlation)


def null_zscore(
    attr: dict,
    causal: dict,
    *,
    n_perm: int = 2000,
    seed: int = 0,
) -> float:
    """Return z-score of real alignment against shuffled-attribution null."""
    _validate_n_perm(n_perm)
    a, c = _aligned_vals(attr, causal)
    real = _spearman(a, c)

    rng = random.Random(seed)
    null = []
    for _ in range(n_perm):
        permuted = a[:]
        rng.shuffle(permuted)
        null.append(_spearman(permuted, c))

    mean = statistics.mean(null)
    sd = statistics.pstdev(null) or 1e-9
    return (real - mean) / sd


def null_pvalue(
    attr: dict,
    causal: dict,
    *,
    n_perm: int = 2000,
    seed: int = 0,
) -> float:
    """Return corrected two-sided permutation p-value for real alignment."""
    _validate_n_perm(n_perm)
    a, c = _aligned_vals(attr, causal)
    real = abs(_spearman(a, c))

    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        permuted = a[:]
        rng.shuffle(permuted)
        if abs(_spearman(permuted, c)) >= real:
            count += 1
    return (count + 1) / (n_perm + 1)


def sample_null_report(
    attr: dict,
    causal: dict,
    *,
    n_perm: int = 2000,
    seed: int = 0,
) -> dict:
    """Return per-sample real alignment, null z-score, and permutation p-value."""
    a, c = _aligned_vals(attr, causal)
    return {
        "spearman": _spearman(a, c),
        "null_zscore": null_zscore(attr, causal, n_perm=n_perm, seed=seed),
        "null_pvalue": null_pvalue(attr, causal, n_perm=n_perm, seed=seed),
        "n_perm": n_perm,
        "seed": seed,
    }


def aggregate_null(
    attr_by_sample: dict,
    causal_by_sample: dict,
    *,
    n_perm: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Aggregate null-model reports across samples in deterministic order."""
    if not attr_by_sample or not causal_by_sample:
        raise ValueError("sample collections must not be empty")
    if set(attr_by_sample) != set(causal_by_sample):
        raise ValueError(
            "attr_by_sample and causal_by_sample must share the same samples"
        )
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    samples = {}
    for index, sample_id in enumerate(sorted(attr_by_sample)):
        samples[sample_id] = sample_null_report(
            attr_by_sample[sample_id],
            causal_by_sample[sample_id],
            n_perm=n_perm,
            seed=seed + index,
        )

    zscores = [report["null_zscore"] for report in samples.values()]
    return {
        "mean_zscore": statistics.mean(zscores),
        "median_zscore": statistics.median(zscores),
        "significant_fraction": (
            sum(report["null_pvalue"] < alpha for report in samples.values())
            / len(samples)
        ),
        "n_samples": len(samples),
        "alpha": alpha,
        "samples": samples,
    }


def _validate_n_perm(n_perm: int) -> None:
    if not isinstance(n_perm, int) or isinstance(n_perm, bool) or n_perm < 1:
        raise ValueError("n_perm must be a positive integer")


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("profile values must be numeric") from exc
