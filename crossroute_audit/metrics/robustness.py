"""Robustness checks for faithfulness metric conclusions."""
from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence

from crossroute_audit.metrics.stats import bootstrap_ci_median
from crossroute_audit.metrics.structure_align import (
    detrended_rank_alignment,
    topk_overlap,
)


def _validate_profile_pair(attr: dict, causal: dict) -> list[str]:
    """Return integer-sorted layer keys after validating two layer profiles."""
    if not attr or not causal:
        raise ValueError("attr and causal must not be empty")
    if set(attr) != set(causal):
        raise ValueError("attr and causal must share the same layers")
    try:
        return sorted(attr, key=lambda key: int(key))
    except (TypeError, ValueError) as exc:
        raise ValueError("layer keys must be integer-like strings") from exc


def topk_curve(attr: dict, causal: dict, ks=None) -> dict:
    """Return top-k overlap for each requested k."""
    layers = _validate_profile_pair(attr, causal)
    layer_count = len(layers)
    requested = list(range(1, layer_count + 1)) if ks is None else list(ks)
    if not requested:
        raise ValueError("ks must not be empty")

    out = {}
    for k in requested:
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= layer_count:
            raise ValueError("k must be between 1 and number of layers")
        out[k] = float(topk_overlap(attr, causal, k=k))
    return out


def _validate_sample_maps(attr_by_sample: dict, causal_by_sample: dict) -> list[str]:
    """Validate sample maps and return sorted sample ids."""
    if not attr_by_sample or not causal_by_sample:
        raise ValueError("sample collections must not be empty")
    if set(attr_by_sample) != set(causal_by_sample):
        raise ValueError(
            "attr_by_sample and causal_by_sample must share the same samples"
        )

    sample_ids = sorted(attr_by_sample)
    reference_layers: set | None = None
    for sample_id in sample_ids:
        layers = _validate_profile_pair(
            attr_by_sample[sample_id],
            causal_by_sample[sample_id],
        )
        layer_set = set(layers)
        if reference_layers is None:
            reference_layers = layer_set
        elif layer_set != reference_layers:
            raise ValueError("all samples must share the same layers")
    return sample_ids


def _layer_profile(sample_profiles: dict, *, reducer: str) -> dict:
    """Return the mean or median layer profile across samples."""
    if reducer not in {"mean", "median"}:
        raise ValueError("reducer must be 'mean' or 'median'")
    if not sample_profiles:
        raise ValueError("sample_profiles must not be empty")

    sample_ids = sorted(sample_profiles)
    layers = _validate_single_profile_collection(sample_profiles)
    profile = {}
    for layer in layers:
        values = [_to_float(sample_profiles[sample_id][layer]) for sample_id in sample_ids]
        if reducer == "mean":
            profile[layer] = statistics.mean(values)
        else:
            profile[layer] = statistics.median(values)
    return profile


def _subtract_profile(sample_profiles: dict, profile: dict) -> dict:
    """Subtract a central profile from each sample profile."""
    if not sample_profiles:
        raise ValueError("sample_profiles must not be empty")
    layers = _validate_single_profile_collection(sample_profiles)
    if set(layers) != set(profile):
        raise ValueError("sample profiles and center profile must share the same layers")

    return {
        sample_id: {
            layer: _to_float(sample_profiles[sample_id][layer]) - _to_float(profile[layer])
            for layer in layers
        }
        for sample_id in sorted(sample_profiles)
    }


def _median_detrended_alignment(
    attr_by_sample: dict,
    causal_by_sample: dict,
    *,
    reducer: str,
) -> float:
    """Return median per-sample alignment after subtracting a center profile."""
    _validate_sample_maps(attr_by_sample, causal_by_sample)
    attr_profile = _layer_profile(attr_by_sample, reducer=reducer)
    causal_profile = _layer_profile(causal_by_sample, reducer=reducer)
    detrended_attr = _subtract_profile(attr_by_sample, attr_profile)
    detrended_causal = _subtract_profile(causal_by_sample, causal_profile)

    result = detrended_rank_alignment(detrended_attr, detrended_causal)
    if not isinstance(result, Mapping):
        raise ValueError("detrended alignment result must contain a scalar score")

    scores = []
    for sample_id in sorted(detrended_attr):
        if sample_id not in result:
            raise ValueError("detrended alignment result must contain a scalar score")
        scores.append(_finite_or_zero(result[sample_id]))
    if not scores:
        raise ValueError("detrended alignment result must contain a scalar score")
    return statistics.median(scores)


def detrend_sensitivity(attr_by_sample: dict, causal_by_sample: dict) -> dict:
    """Compare median detrended alignment under mean vs median profiles."""
    mean_alignment = _median_detrended_alignment(
        attr_by_sample,
        causal_by_sample,
        reducer="mean",
    )
    median_alignment = _median_detrended_alignment(
        attr_by_sample,
        causal_by_sample,
        reducer="median",
    )
    return {
        "mean_profile_median_alignment": mean_alignment,
        "median_profile_median_alignment": median_alignment,
        "absolute_difference": abs(mean_alignment - median_alignment),
    }


def bootstrap_stability(
    values,
    seeds=(0, 1, 2, 3, 4),
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """Return percentile-bootstrap median CI stability across seeds."""
    prepared_values = _validate_values(values)
    prepared_seeds = list(seeds)
    if not prepared_seeds:
        raise ValueError("seeds must not be empty")
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    ci_by_seed = {}
    width_by_seed = {}
    for seed in prepared_seeds:
        lo, hi = _bootstrap_ci(
            prepared_values,
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
        )
        ci_by_seed[seed] = [lo, hi]
        width_by_seed[seed] = hi - lo

    widths = list(width_by_seed.values())
    return {
        "median": statistics.median(prepared_values),
        "ci_by_seed": ci_by_seed,
        "width_by_seed": width_by_seed,
        "mean_width": statistics.mean(widths),
        "max_width": max(widths),
        "min_width": min(widths),
    }


def robustness_report(
    attr: dict,
    causal: dict,
    attr_by_sample: dict,
    causal_by_sample: dict,
    values,
    *,
    ks=None,
    seeds=(0, 1, 2, 3, 4),
    stable_threshold: float = 0.05,
) -> dict:
    """Return a combined robustness battery without drawing causal conclusions."""
    if stable_threshold < 0:
        raise ValueError("stable_threshold must be non-negative")

    topk = topk_curve(attr, causal, ks=ks)
    detrend = detrend_sensitivity(attr_by_sample, causal_by_sample)
    bootstrap = bootstrap_stability(values, seeds=seeds)
    return {
        "topk_curve": topk,
        "detrend_sensitivity": detrend,
        "bootstrap_stability": bootstrap,
        "stable": detrend["absolute_difference"] <= stable_threshold,
    }


def _validate_single_profile_collection(sample_profiles: dict) -> list:
    sample_ids = sorted(sample_profiles)
    first = sample_profiles[sample_ids[0]]
    layers = _validate_profile_pair(first, first)
    reference = set(layers)
    for sample_id in sample_ids[1:]:
        current = _validate_profile_pair(sample_profiles[sample_id], sample_profiles[sample_id])
        if set(current) != reference:
            raise ValueError("all samples must share the same layers")
    return layers


def _bootstrap_ci(
    values: list[float],
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    try:
        lo, hi = bootstrap_ci_median(
            values,
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
        )
        return float(lo), float(hi)
    except TypeError:
        return _fallback_bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)


def _fallback_bootstrap_ci(
    values: list[float],
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choice(values) for _ in range(len(values)))
        for _ in range(n_boot)
    )
    return (
        _percentile(medians, alpha / 2),
        _percentile(medians, 1 - alpha / 2),
    )


def _percentile(sorted_values: list[float], quantile: float) -> float:
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] + weight * (sorted_values[upper] - sorted_values[lower]))


def _validate_values(values) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        values = list(values) if isinstance(values, Iterable) else []
    if not values:
        raise ValueError("values must not be empty")
    return [_to_float(value) for value in values]


def _to_float(value) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("values must be finite")
    return numeric


def _finite_or_zero(value) -> float:
    if value is None:
        return 0.0
    numeric = float(value)
    return numeric if math.isfinite(numeric) else 0.0
