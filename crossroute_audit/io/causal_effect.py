"""Build and write causal-effect artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from crossroute_audit.io.manifest import load_manifest
from crossroute_audit.metrics.causal_effect import (
    causal_effect_by_layer,
    effect_stability,
)


DEFAULT_GROUPS = ("image", "text")


def _normalize_groups(groups: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(groups, str):
        names = tuple(name.strip() for name in groups.split(",") if name.strip())
    else:
        names = tuple(str(name).strip() for name in groups if str(name).strip())
    if not names:
        raise ValueError("groups must contain at least one group name")
    if len(set(names)) != len(names):
        raise ValueError("groups must not contain duplicates")
    return names


def _layer_map_for_json(effects: dict) -> dict[str, float]:
    """Convert numeric layer keys to stable JSON object keys."""
    return {str(layer): float(effects[layer]) for layer in sorted(effects)}


def build_causal_effect_result(
    adapter,
    inputs,
    sample,
    groups=DEFAULT_GROUPS,
    mode: str = "ablate",
    stability_layer: int = 0,
    stability_repeats: int = 3,
) -> dict:
    """Build the per-sample causal-effect artifact payload."""
    group_names = _normalize_groups(groups)
    layer_count = int(adapter.get_intervention_layer_count())
    if layer_count <= 0:
        raise ValueError(f"intervention layer count must be positive, got {layer_count}")
    if not isinstance(stability_layer, int):
        raise TypeError("stability_layer must be an integer")
    if stability_layer < 0 or stability_layer >= layer_count:
        raise IndexError(
            f"stability_layer {stability_layer} is outside valid range [0, {layer_count - 1}]"
        )

    c_by_layer = {}
    stability = {}
    for group in group_names:
        c_by_layer[group] = _layer_map_for_json(
            causal_effect_by_layer(adapter, inputs, sample, group=group, mode=mode)
        )
        stability[group] = {
            str(stability_layer): effect_stability(
                adapter,
                inputs,
                sample,
                layer=stability_layer,
                group=group,
                mode=mode,
                repeats=stability_repeats,
            )
        }

    return {
        "sample_id": sample["sample_id"],
        "target_answer": sample["target_answer"],
        "C_by_layer": c_by_layer,
        "stability": stability,
        "settings": {
            "mode": mode,
            "layer_count": layer_count,
            "groups": list(group_names),
            "stability_layer": stability_layer,
            "stability_repeats": stability_repeats,
        },
    }


def write_causal_effect(result: dict, out_path) -> None:
    """Write a deterministic JSON causal-effect artifact."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_file:
        json.dump(result, out_file, sort_keys=True, indent=2, ensure_ascii=False)
        out_file.write("\n")


def causal_effect_for_manifest(
    adapter,
    manifest_path,
    out_dir,
    groups=DEFAULT_GROUPS,
    mode: str = "ablate",
) -> list[str]:
    """Build one causal-effect JSON file per manifest sample."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sample in load_manifest(manifest_path):
        inputs = adapter.prepare_inputs(sample["image_path"], sample["question"])
        result = build_causal_effect_result(
            adapter,
            inputs,
            sample,
            groups=groups,
            mode=mode,
        )
        out_path = output_dir / f"causal_effect_{sample['sample_id']}.json"
        write_causal_effect(result, out_path)
        paths.append(str(out_path))
    return paths
