"""Aggregate per-token attribution into image/text mass by LM layer."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


_GROUP_NAMES = ("image", "text")


def attribution_mass_for_layer(
    token_attribution: Sequence[float],
    token_groups,
) -> dict[str, float]:
    """Return ``sum(abs(token attribution))`` for image and text positions."""
    values = tuple(float(value) for value in token_attribution)
    if not values:
        raise ValueError("token_attribution must not be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("token_attribution values must be finite")

    result = {}
    for group_name in _GROUP_NAMES:
        positions = tuple(int(position) for position in getattr(token_groups, group_name))
        if not positions:
            raise ValueError(f"token group {group_name!r} must not be empty")
        if len(set(positions)) != len(positions):
            raise ValueError(f"token group {group_name!r} contains duplicate positions")
        if min(positions) < 0 or max(positions) >= len(values):
            raise IndexError(
                f"token group {group_name!r} references positions outside "
                f"the attribution length {len(values)}"
            )
        result[group_name] = float(sum(abs(values[position]) for position in positions))
    return result


def attribution_mass_by_layer(
    token_attribution_by_layer: Mapping[int, Sequence[float]],
    token_groups,
    layer_count: int,
) -> dict[str, dict[int, float]]:
    """Aggregate attribution on the causal LM-encoder layer axis.

    ``token_attribution_by_layer`` must contain every zero-based layer in
    ``range(layer_count)``. Requiring the full axis prevents attribution and
    causal-effect artifacts from being compared after a silent layer mismatch.
    """
    if not isinstance(layer_count, int):
        raise TypeError("layer_count must be an integer")
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")

    expected_layers = set(range(layer_count))
    actual_layers = set(token_attribution_by_layer)
    if actual_layers != expected_layers:
        missing = sorted(expected_layers - actual_layers)
        extra = sorted(actual_layers - expected_layers)
        raise ValueError(
            "token attribution layer axis mismatch: "
            f"missing={missing}, extra={extra}"
        )

    output = {group_name: {} for group_name in _GROUP_NAMES}
    for layer in range(layer_count):
        layer_mass = attribution_mass_for_layer(
            token_attribution_by_layer[layer],
            token_groups,
        )
        for group_name in _GROUP_NAMES:
            output[group_name][layer] = layer_mass[group_name]
    return output


def attribution_mass_for_manifest(
    adapter,
    manifest_path,
    out_dir,
    *,
    n_steps: int | None = None,
    internal_batch_size: int = 1,
) -> list[str]:
    """Compute and write AttributionMass artifacts for a manifest.

    The import is intentionally local: aggregation stays torch-free and the
    Captum dependency is loaded only when attribution computation is requested.
    """
    from crossroute_audit.attribution.integrated_gradients import (
        attribution_mass_for_manifest as _write_manifest_attribution,
    )

    return _write_manifest_attribution(
        adapter,
        manifest_path,
        out_dir,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )
