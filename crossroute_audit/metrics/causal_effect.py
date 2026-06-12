"""Causal-effect metrics over the intervention layer axis."""
from __future__ import annotations

import math


def _as_finite_float(value, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar!r}")
    return scalar


def _intervention_layer_count(adapter) -> int:
    if not hasattr(adapter, "get_intervention_layer_count"):
        raise AttributeError(
            "adapter must expose get_intervention_layer_count(); do not use get_layer_count() "
            "for causal-effect metrics"
        )
    layer_count = int(adapter.get_intervention_layer_count())
    if layer_count <= 0:
        raise ValueError(f"intervention layer count must be positive, got {layer_count}")
    return layer_count


def _validate_layer(adapter, layer: int) -> int:
    if not isinstance(layer, int):
        raise TypeError("layer must be an integer")
    layer_count = _intervention_layer_count(adapter)
    if layer < 0 or layer >= layer_count:
        raise IndexError(f"layer {layer} is outside valid range [0, {layer_count - 1}]")
    return layer_count


def causal_effect(clean_logit: float, intervened_logit: float) -> float:
    """Return the scalar causal effect: clean target logit minus intervened logit."""
    clean = _as_finite_float(clean_logit, "clean_logit")
    intervened = _as_finite_float(intervened_logit, "intervened_logit")
    return clean - intervened


def causal_effect_by_layer(adapter, inputs, sample, group: str, mode: str = "ablate") -> dict:
    """Return ``{layer_index: effect}`` for every LM-encoder intervention layer.

    The target logit is resolved on the same ``inputs`` before interventions so
    clean and intervened logits refer to the same target token.
    """
    if not group:
        raise ValueError("group must be a non-empty string")
    if not mode:
        raise ValueError("mode must be a non-empty string")

    layer_count = _intervention_layer_count(adapter)
    clean = adapter.get_target_logit(
        inputs,
        sample["target_answer"],
        sample["target_token_policy"],
    )
    effects = {}
    for layer in range(layer_count):
        intervened = adapter.intervene(inputs, layer, group, mode)
        effects[layer] = causal_effect(clean, intervened)
    return effects


def effect_stability(
    adapter,
    inputs,
    sample,
    layer: int,
    group: str,
    mode: str = "ablate",
    repeats: int = 3,
) -> dict:
    """Repeat one intervention and summarize effect stability at one layer."""
    if not isinstance(repeats, int):
        raise TypeError("repeats must be an integer")
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if not group:
        raise ValueError("group must be a non-empty string")
    if not mode:
        raise ValueError("mode must be a non-empty string")
    _validate_layer(adapter, layer)

    effects = []
    for _ in range(repeats):
        clean = adapter.get_target_logit(
            inputs,
            sample["target_answer"],
            sample["target_token_policy"],
        )
        intervened = adapter.intervene(inputs, layer, group, mode)
        effects.append(causal_effect(clean, intervened))

    mean = sum(effects) / len(effects)
    variance = sum((effect - mean) ** 2 for effect in effects) / len(effects)
    std = math.sqrt(variance)
    max_abs_dev = max(abs(effect - mean) for effect in effects)
    return {
        "mean": mean,
        "std": std,
        "max_abs_dev": max_abs_dev,
    }
