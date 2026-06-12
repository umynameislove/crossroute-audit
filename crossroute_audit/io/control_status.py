"""Build and write control-status artifacts for baseline gates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from crossroute_audit.controls import baselines
from crossroute_audit.io.manifest import load_manifest


DEFAULT_CONTROLS = ("text_only", "no_image", "counterfactual", "negative_control")
SUPPORTED_CONTROLS = frozenset(DEFAULT_CONTROLS)


def _normalize_which(which: Iterable[str] | str | None) -> tuple[str, ...]:
    if which is None:
        names = DEFAULT_CONTROLS
    elif isinstance(which, str):
        names = tuple(name.strip() for name in which.split(",") if name.strip())
    else:
        names = tuple(str(name).strip() for name in which if str(name).strip())
    if not names:
        raise ValueError("which must contain at least one control name")
    unknown = sorted(set(names) - SUPPORTED_CONTROLS)
    if unknown:
        supported = ", ".join(DEFAULT_CONTROLS)
        raise ValueError(
            f"unsupported control name(s): {', '.join(unknown)}; expected one of {supported}"
        )
    return names


def build_control_status(
    adapter,
    sample,
    which=DEFAULT_CONTROLS,
) -> dict:
    """Run baseline controls for one sample and wrap them with stable metadata."""
    names = _normalize_which(which)
    controls = baselines.run_controls(adapter, sample, which=",".join(names))
    missing = [name for name in names if name not in controls]
    if missing:
        raise RuntimeError(
            f"run_controls did not return requested control(s): {', '.join(missing)}"
        )
    return {
        "sample_id": sample["sample_id"],
        "target_answer": sample["target_answer"],
        "which": list(names),
        "controls": controls,
    }


def write_control_status(status: dict, out_path) -> None:
    """Write a deterministic JSON control-status artifact."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_file:
        json.dump(status, out_file, sort_keys=True, indent=2, ensure_ascii=False)
        out_file.write("\n")


def control_status_for_manifest(adapter, manifest_path, out_dir) -> list[str]:
    """Build one control-status JSON file per manifest sample."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sample in load_manifest(manifest_path):
        status = build_control_status(adapter, sample)
        out_path = output_dir / f"control_status_{sample['sample_id']}.json"
        write_control_status(status, out_path)
        paths.append(str(out_path))
    return paths
