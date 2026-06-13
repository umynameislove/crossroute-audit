"""Secondary diagnostics: normalized attribution-flow gap and flow retention."""
from __future__ import annotations

import statistics


def _zscore(values):
    mean = statistics.mean(values)
    std = statistics.pstdev(values) or 1.0
    return [(value - mean) / std for value in values]


def attribution_flow_gap(attr_by_layer: dict, causal_by_layer: dict) -> dict:
    """Return |A_l - C_l| after z-score normalization over common layers."""
    layers = sorted(set(attr_by_layer) & set(causal_by_layer))
    if not layers:
        return {}

    attr_z = _zscore([attr_by_layer[layer] for layer in layers])
    causal_z = _zscore([causal_by_layer[layer] for layer in layers])
    return {
        layer: abs(attr - causal)
        for layer, attr, causal in zip(layers, attr_z, causal_z)
    }


def flow_retention(causal_image_by_layer: dict) -> dict:
    """Return C_image_l / C_image(first layer), preserving the original layer keys."""
    if not causal_image_by_layer:
        return {}

    first = causal_image_by_layer[min(causal_image_by_layer)]
    base = first if first != 0 else 1.0
    return {layer: value / base for layer, value in causal_image_by_layer.items()}
