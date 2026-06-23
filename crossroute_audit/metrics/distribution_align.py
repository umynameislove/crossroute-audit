"""Distribution-level alignment metrics for layer profiles."""
from __future__ import annotations

import math
from typing import Any


def _prob(series: dict) -> tuple[list[Any], list[float]]:
    """Return sorted layer keys and non-negative normalized probabilities."""
    layers = _sorted_layers(series)
    values = [_clamped_value(series[layer]) for layer in layers]
    total = sum(values)
    if total <= 0:
        raise ValueError("profile must have positive mass")
    return layers, [value / total for value in values]


def profile_emd(attr: dict, causal: dict) -> float:
    """Return 1D Earth Mover's Distance between normalized layer profiles."""
    attr_layers, attr_prob = _prob(attr)
    causal_layers, causal_prob = _prob(causal)
    _require_same_layers(attr_layers, causal_layers)

    attr_cdf = 0.0
    causal_cdf = 0.0
    distance = 0.0
    for attr_value, causal_value in zip(attr_prob, causal_prob):
        attr_cdf += attr_value
        causal_cdf += causal_value
        distance += abs(attr_cdf - causal_cdf)
    return distance


def center_of_mass_shift(attr: dict, causal: dict) -> float:
    """Return COM(attr) - COM(causal), using sorted layer positions."""
    attr_layers, attr_prob = _prob(attr)
    causal_layers, causal_prob = _prob(causal)
    _require_same_layers(attr_layers, causal_layers)

    def center_of_mass(probabilities: list[float]) -> float:
        return sum(index * value for index, value in enumerate(probabilities))

    return center_of_mass(attr_prob) - center_of_mass(causal_prob)


def normalized_emd(attr: dict, causal: dict) -> float:
    """Return EMD divided by the maximum possible distance for the layer count."""
    attr_layers = _sorted_layers(attr)
    if len(attr_layers) < 2:
        raise ValueError("normalized_emd requires at least two layers")
    return profile_emd(attr, causal) / (len(attr_layers) - 1)


def _sorted_layers(series: dict) -> list[Any]:
    if not series:
        raise ValueError("profile must not be empty")

    parsed_layers = [(_parse_layer_key(layer), layer) for layer in series]
    parsed_ids = [parsed for parsed, _ in parsed_layers]
    if len(set(parsed_ids)) != len(parsed_ids):
        raise ValueError("layer keys must map to unique integer layers")
    return [layer for _, layer in sorted(parsed_layers, key=lambda item: item[0])]


def _parse_layer_key(layer: Any) -> int:
    try:
        return int(layer)
    except (TypeError, ValueError) as exc:
        raise ValueError("layer keys must be parseable as integers") from exc


def _clamped_value(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("profile values must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError("profile values must be finite")
    return max(numeric, 0.0)


def _require_same_layers(left: list[Any], right: list[Any]) -> None:
    if [_parse_layer_key(layer) for layer in left] != [
        _parse_layer_key(layer) for layer in right
    ]:
        raise ValueError("attr and causal must share the same layers")
