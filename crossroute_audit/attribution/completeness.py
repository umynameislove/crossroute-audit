"""Completeness checks for Integrated Gradients attribution."""
from __future__ import annotations

import math


def completeness_residual(
    attribution_sum: float,
    target_logit: float,
    baseline_logit: float,
    *,
    epsilon: float = 1e-8,
) -> float:
    """Return the normalized absolute IG completeness residual.

    Integrated Gradients satisfies completeness when the sum of all feature
    attributions equals ``target_logit - baseline_logit``. The residual is
    normalized by the absolute output difference so it is comparable across
    samples and layers. A large value indicates poor numerical convergence and
    should lower confidence downstream.
    """
    values = {
        "attribution_sum": float(attribution_sum),
        "target_logit": float(target_logit),
        "baseline_logit": float(baseline_logit),
        "epsilon": float(epsilon),
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if values["epsilon"] <= 0:
        raise ValueError("epsilon must be positive")

    output_delta = values["target_logit"] - values["baseline_logit"]
    absolute_error = abs(values["attribution_sum"] - output_delta)
    return absolute_error / max(abs(output_delta), values["epsilon"])


def convergence_delta_residual(
    convergence_delta: float,
    target_logit: float,
    baseline_logit: float,
    *,
    epsilon: float = 1e-8,
) -> float:
    """Normalize a Captum convergence delta into the project residual scale."""
    delta = float(convergence_delta)
    target = float(target_logit)
    baseline = float(baseline_logit)
    eps = float(epsilon)
    if not all(math.isfinite(value) for value in (delta, target, baseline, eps)):
        raise ValueError("convergence delta inputs must be finite")
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    return abs(delta) / max(abs(target - baseline), eps)
