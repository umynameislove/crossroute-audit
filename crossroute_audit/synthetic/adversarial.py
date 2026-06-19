"""Noisy synthetic cases, degradation curves, and rank-based ROC AUC."""
from __future__ import annotations

import math
import random

from crossroute_audit.metrics.diagnosis import diagnose
from crossroute_audit.synthetic.benchmark import FAULT_CLASSES, generate_case


def noisy_case(kind, rng, sigma):
    """Generate a benchmark case and add Gaussian noise to numeric evidence."""
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be a non-negative finite number")

    case = generate_case(kind, rng)
    if sigma == 0:
        return case

    control, causal, attribution, rank, routing, expected = case
    return (
        _add_noise(control, rng, sigma),
        _add_noise(causal, rng, sigma),
        _add_noise(attribution, rng, sigma),
        _add_noise(rank, rng, sigma),
        _add_noise(routing, rng, sigma),
        expected,
    )


def degradation_curve(
    sigmas: list[float],
    n_per_fault: int = 30,
    seed: int = 0,
) -> list[dict]:
    """Return detector accuracy at each requested Gaussian noise level."""
    if not isinstance(n_per_fault, int) or isinstance(n_per_fault, bool):
        raise ValueError("n_per_fault must be a positive integer")
    if n_per_fault < 1:
        raise ValueError("n_per_fault must be a positive integer")
    if any(not math.isfinite(sigma) or sigma < 0 for sigma in sigmas):
        raise ValueError("sigmas must contain non-negative finite numbers")

    curve = []
    for sigma in sigmas:
        correct = 0
        total = 0
        for kind in FAULT_CLASSES:
            for case_index in range(n_per_fault):
                rng = random.Random((seed, kind, case_index).__repr__())
                (
                    control,
                    causal,
                    attribution,
                    rank,
                    routing,
                    expected,
                ) = noisy_case(kind, rng, sigma)
                got = diagnose(
                    control,
                    causal,
                    attribution,
                    rank,
                    routing_proxy=routing,
                )["diagnosis"]
                correct += got == expected
                total += 1
        curve.append({"sigma": sigma, "accuracy": correct / total})
    return curve


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Return Mann-Whitney ROC AUC with average ranks for tied scores."""
    if not scores or not labels:
        raise ValueError("scores and labels must not be empty")
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("labels must contain only 0 and 1")

    positive_count = sum(label == 1 for label in labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("labels must contain both classes")

    ranked = sorted(
        enumerate(scores),
        key=lambda item: (item[1], item[0]),
    )
    rank_by_index = [0.0] * len(scores)
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][1] == ranked[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        for original_index, _ in ranked[start:end]:
            rank_by_index[original_index] = average_rank
        start = end

    positive_rank_sum = sum(
        rank_by_index[index]
        for index, label in enumerate(labels)
        if label == 1
    )
    u_statistic = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2
    )
    return u_statistic / (positive_count * negative_count)


def _add_noise(value, rng, sigma):
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            key: _add_noise(nested_value, rng, sigma)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_add_noise(item, rng, sigma) for item in value]
    if isinstance(value, tuple):
        return tuple(_add_noise(item, rng, sigma) for item in value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value + rng.gauss(0.0, sigma)
    return value
