"""RankAlignment: Spearman correlation between attribution and causal effect."""
from __future__ import annotations

from scipy.stats import spearmanr


def rank_alignment(attribution_by_layer: dict, causal_by_layer: dict) -> float | None:
    """Return Spearman rho over common layers, or None when it is undefined."""
    layers = sorted(set(attribution_by_layer) & set(causal_by_layer))
    if len(layers) < 2:
        return None

    attribution_values = [attribution_by_layer[layer] for layer in layers]
    causal_values = [causal_by_layer[layer] for layer in layers]
    if len(set(attribution_values)) < 2 or len(set(causal_values)) < 2:
        return None

    rho, _ = spearmanr(attribution_values, causal_values)
    return float(rho)


def rank_alignment_by_group(attribution_mass: dict, causal_effect: dict) -> dict:
    """Return {group: rho} for groups present in both attribution and causal maps."""
    out = {}
    for group in sorted(set(attribution_mass) & set(causal_effect)):
        out[group] = rank_alignment(attribution_mass[group], causal_effect[group])
    return out
