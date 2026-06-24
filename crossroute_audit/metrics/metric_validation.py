"""Synthetic recovery checks for CrossRoute faithfulness metrics."""
from __future__ import annotations

import math
import random
from numbers import Real
from typing import Iterable

from crossroute_audit.metrics.distribution_align import profile_emd
from crossroute_audit.metrics.rank_alignment import rank_alignment_by_group
from crossroute_audit.metrics.structure_align import topk_overlap


def make_pair(
    kind: str,
    n_layers: int,
    rng,
    *,
    level: float = 1.0,
):
    """Return (attr, causal) as {layer_str: value} with known relationship."""
    if not isinstance(n_layers, int) or isinstance(n_layers, bool) or n_layers < 2:
        raise ValueError("n_layers must be at least 2")
    if level < 0:
        raise ValueError("level must be non-negative")

    causal = [rng.uniform(0.0, 10.0) for _ in range(n_layers)]
    if kind == "faithful":
        attr = [c + rng.gauss(0, 0.01) for c in causal]
    elif kind == "anti":
        attr = [-c for c in causal]
    elif kind == "shuffled":
        attr = causal[:]
        rng.shuffle(attr)
    elif kind == "noisy":
        attr = [c + rng.gauss(0, level) for c in causal]
    else:
        raise ValueError(f"unknown kind {kind!r}")

    return _to_layer_map(attr), _to_layer_map(causal)


def recovery_report(
    n_layers: int = 24,
    seed: int = 0,
    *,
    levels=(0.5, 3.0, 10.0),
    topk: int | None = None,
) -> dict:
    """Return metric recovery results for synthetic known relationships."""
    if not levels:
        raise ValueError("levels must not be empty")
    effective_topk = max(1, n_layers // 4) if topk is None else topk

    rng = random.Random(seed)
    report = {}
    for kind in ("faithful", "anti", "shuffled"):
        attr, causal = make_pair(kind, n_layers, rng)
        report[kind] = _metric_bundle(attr, causal, topk=effective_topk)

    noisy_by_level = {}
    for level in levels:
        attr, causal = make_pair("noisy", n_layers, rng, level=level)
        noisy_by_level[level] = _metric_bundle(attr, causal, topk=effective_topk)
    report["noisy_by_level"] = noisy_by_level
    return report


def _metric_bundle(attr: dict, causal: dict, *, topk: int) -> dict:
    """Return scalar rank, top-k, and distributional metrics for one pair."""
    return {
        "rank_alignment": _scalar_rank_alignment(attr, causal),
        "topk_overlap": topk_overlap(attr, causal, k=topk),
        "profile_emd": _safe_profile_emd(attr, causal),
    }


def _scalar_rank_alignment(attr: dict, causal: dict) -> float:
    """Return the scalar synthetic-group rank alignment."""
    result = rank_alignment_by_group(
        {"synthetic": attr},
        {"synthetic": causal},
    )
    score = result.get("synthetic")
    if (
        score is None
        or isinstance(score, bool)
        or not isinstance(score, Real)
        or not math.isfinite(float(score))
    ):
        raise ValueError("rank alignment result must contain a scalar score")
    return float(score)


def _safe_profile_emd(attr: dict, causal: dict) -> float:
    """Return profile EMD, or infinity when the profile has no positive mass."""
    try:
        return float(profile_emd(attr, causal))
    except ValueError as exc:
        if "positive mass" not in str(exc):
            raise
        return float("inf")


def _to_layer_map(values: Iterable[float]) -> dict[str, float]:
    return {str(i): value for i, value in enumerate(values)}
