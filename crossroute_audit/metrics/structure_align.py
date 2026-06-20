"""Confound-robust structural alignment metrics for attribution vs causal.

Scalar RankAlignment is confounded by a shared front-loaded layer profile.
These two metrics isolate sample-specific, structure-level faithfulness:
detrended rank alignment removes the cross-sample mean layer profile; top-k
overlap checks whether attribution's strongest layers match causal's.
"""
from __future__ import annotations

import math

from scipy.stats import spearmanr


def _layers(series: dict) -> list[str]:
    if not series:
        raise ValueError("layer series must not be empty")
    return sorted(series, key=lambda layer: int(layer))


def detrended_rank_alignment(
    attr_by_sample: dict,
    causal_by_sample: dict,
) -> dict:
    """Per-sample Spearman rho after removing the cross-sample mean layer profile.

    Both inputs are ``{sample_id: {layer: value}}`` for one route (e.g. image),
    sharing the same sample ids and layer keys. Returns ``{sample_id: rho}`` where
    rho is ``None`` when a detrended series is constant (rho undefined).
    """
    sids = list(attr_by_sample)
    if not sids or set(sids) != set(causal_by_sample):
        raise ValueError("attr and causal must share the same sample ids")
    layers = _layers(attr_by_sample[sids[0]])
    for s in sids:
        if set(attr_by_sample[s]) != set(layers) or set(causal_by_sample[s]) != set(layers):
            raise ValueError(f"sample {s!r} has inconsistent layer keys")

    mean_a = {L: sum(attr_by_sample[s][L] for s in sids) / len(sids) for L in layers}
    mean_c = {L: sum(causal_by_sample[s][L] for s in sids) / len(sids) for L in layers}

    out = {}
    for s in sids:
        da = [attr_by_sample[s][L] - mean_a[L] for L in layers]
        dc = [causal_by_sample[s][L] - mean_c[L] for L in layers]
        rho = spearmanr(da, dc).correlation
        out[s] = None if rho is None or math.isnan(rho) else float(rho)
    return out


def topk_overlap(attr: dict, causal: dict, k: int = 5) -> float:
    """Fraction of the top-k attribution layers that are also top-k causal layers."""
    layers = _layers(attr)
    if set(layers) != set(causal):
        raise ValueError("attr and causal must share the same layers")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > len(layers):
        raise ValueError(f"k must be an integer in [1, {len(layers)}]")
    top_a = set(sorted(attr, key=lambda L: attr[L], reverse=True)[:k])
    top_c = set(sorted(causal, key=lambda L: causal[L], reverse=True)[:k])
    return len(top_a & top_c) / k
