"""Wilcoxon-based simulation utilities for sample-size power analysis."""
from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence

from scipy.stats import wilcoxon


def _validate_values(values) -> list[float]:
    """Return a numeric copy of values."""
    try:
        raw_values = list(values)
    except TypeError as exc:
        raise ValueError("values must not be empty") from exc
    if not raw_values:
        raise ValueError("values must not be empty")
    converted = []
    for value in raw_values:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("values must be numeric") from exc
        if not math.isfinite(numeric):
            raise ValueError("values must be numeric")
        converted.append(numeric)
    return converted


def _validate_sizes(sizes) -> list[int]:
    """Return validated positive integer sample sizes."""
    try:
        raw_sizes = list(sizes)
    except TypeError as exc:
        raise ValueError("sizes must not be empty") from exc
    if not raw_sizes:
        raise ValueError("sizes must not be empty")
    validated = []
    for size in raw_sizes:
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("sizes must be positive integers")
        validated.append(size)
    return validated


def _wilcoxon_pvalue(sample: list[float]) -> float:
    """Return a safe one-sample Wilcoxon p-value against zero."""
    if all(value == 0 for value in sample):
        return 1.0
    try:
        pvalue = wilcoxon(sample).pvalue
    except ValueError:
        return 1.0
    if pvalue is None or not math.isfinite(float(pvalue)):
        return 1.0
    return float(pvalue)


def power_curve(
    values,
    sizes=(20, 50, 100, 200),
    *,
    alpha=0.05,
    n_sim=1000,
    seed=0,
) -> dict:
    """Estimate Wilcoxon rejection probability for each candidate sample size."""
    prepared_values = _validate_values(values)
    prepared_sizes = _validate_sizes(sizes)
    _validate_alpha(alpha)
    _validate_n_sim(n_sim)

    rng = random.Random(seed)
    out = {}
    for sample_size in prepared_sizes:
        hits = 0
        for _ in range(n_sim):
            sample = [
                prepared_values[rng.randrange(len(prepared_values))]
                for _ in range(sample_size)
            ]
            if any(value != 0 for value in sample) and _wilcoxon_pvalue(sample) < alpha:
                hits += 1
        out[sample_size] = hits / n_sim
    return out


def min_n_for_power(
    values,
    target=0.8,
    sizes=(20, 50, 100, 200),
    *,
    alpha=0.05,
    n_sim=1000,
    seed=0,
) -> int | None:
    """Return the smallest candidate N whose simulated power reaches target."""
    if not 0 < target < 1:
        raise ValueError("target must be between 0 and 1")
    curve = power_curve(
        values,
        sizes=sizes,
        alpha=alpha,
        n_sim=n_sim,
        seed=seed,
    )
    for sample_size in curve:
        if curve[sample_size] >= target:
            return sample_size
    return None


def power_table(
    metric_values_by_model: dict,
    sizes=(20, 50, 100, 200),
    *,
    target=0.8,
    alpha=0.05,
    n_sim=1000,
    seed=0,
) -> dict:
    """Return simulated power curves and minimum N for each model/metric."""
    if not metric_values_by_model:
        raise ValueError("metric_values_by_model must not be empty")
    if not isinstance(metric_values_by_model, Mapping):
        raise ValueError("metric_values_by_model must not be empty")
    _min_from_curve({}, target=target)

    table = {}
    for model_index, model in enumerate(sorted(metric_values_by_model)):
        metrics = metric_values_by_model[model]
        if not metrics or not isinstance(metrics, Mapping):
            raise ValueError("model metrics must not be empty")

        table[model] = {}
        for metric_index, metric in enumerate(sorted(metrics)):
            values = _validate_values(metrics[metric])
            metric_seed = seed + model_index * 1000 + metric_index
            curve = power_curve(
                values,
                sizes=sizes,
                alpha=alpha,
                n_sim=n_sim,
                seed=metric_seed,
            )
            table[model][metric] = {
                "power_curve": curve,
                "min_n_for_target": _min_from_curve(curve, target=target),
                "target": target,
                "alpha": alpha,
            }
    return table


def summarize_power_table(table: dict) -> list[dict]:
    """Flatten a nested power table into deterministic report rows."""
    rows = []
    for model in sorted(table):
        for metric in sorted(table[model]):
            entry = table[model][metric]
            curve = entry["power_curve"]
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "min_n_for_target": entry["min_n_for_target"],
                    "target": entry["target"],
                    "alpha": entry["alpha"],
                    "max_power": max(curve.values()) if curve else 0.0,
                }
            )
    return rows


def _min_from_curve(curve: dict, *, target: float) -> int | None:
    if not 0 < target < 1:
        raise ValueError("target must be between 0 and 1")
    for sample_size in curve:
        if curve[sample_size] >= target:
            return sample_size
    return None


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")


def _validate_n_sim(n_sim: int) -> None:
    if not isinstance(n_sim, int) or isinstance(n_sim, bool) or n_sim <= 0:
        raise ValueError("n_sim must be a positive integer")
