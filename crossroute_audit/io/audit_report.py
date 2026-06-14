"""Build and write the per-sample audit report (final GĐ4 artifact)."""
from __future__ import annotations

import json
from pathlib import Path

from crossroute_audit.io.manifest import load_manifest
from crossroute_audit.metrics.diagnosis import diagnose
from crossroute_audit.metrics.flow_diagnostics import (
    attribution_flow_gap,
    flow_retention,
)
from crossroute_audit.metrics.rank_alignment import rank_alignment_by_group


def build_audit_report(
    sample,
    control_status,
    causal_effect,
    attribution,
    rank_alignment,
    secondary,
    diagnosis,
    settings,
) -> dict:
    """Build the compact per-sample audit report payload."""
    return {
        "sample_id": sample["sample_id"],
        "target_answer": sample["target_answer"],
        "rank_alignment": rank_alignment,
        "secondary": secondary,
        "control_status": control_status.get("controls"),
        "diagnosis": diagnosis,
        "settings": settings,
    }


def write_audit_report(report: dict, out_path) -> None:
    """Write a deterministic JSON audit-report artifact."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_file:
        json.dump(report, out_file, sort_keys=True, indent=2, ensure_ascii=False)
        out_file.write("\n")


def _load_json(directory, prefix, sample_id):
    """Load one named artifact for a sample."""
    path = Path(directory) / f"{prefix}_{sample_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def audit_report_for_manifest(
    manifest_path,
    control_dir,
    causal_dir,
    attribution_dir,
    out_dir,
    attr_thresh=1.0,
    causal_thresh=0.5,
    align_thresh=0.0,
) -> list[str]:
    """Combine per-sample artifacts into final audit-report JSON files."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for sample in load_manifest(manifest_path):
        sample_id = sample["sample_id"]
        control_status = _load_json(control_dir, "control_status", sample_id)
        causal = _load_json(causal_dir, "causal_effect", sample_id)["C_by_layer"]
        attribution = _load_json(
            attribution_dir,
            "attribution_mass",
            sample_id,
        )["attribution_mass"]

        rank = rank_alignment_by_group(attribution, causal)
        secondary = {
            "attribution_flow_gap": {
                group: attribution_flow_gap(attribution[group], causal[group])
                for group in attribution
                if group in causal
            },
            "flow_retention": flow_retention(causal.get("image", {})),
        }
        diag = diagnose(
            control_status,
            causal,
            attribution,
            rank,
            attr_thresh=attr_thresh,
            causal_thresh=causal_thresh,
            align_thresh=align_thresh,
        )
        report = build_audit_report(
            sample,
            control_status,
            causal,
            attribution,
            rank,
            secondary,
            diag,
            settings={
                "attr_thresh": attr_thresh,
                "causal_thresh": causal_thresh,
                "align_thresh": align_thresh,
            },
        )

        out_path = output_dir / f"audit_report_{sample_id}.json"
        write_audit_report(report, out_path)
        paths.append(str(out_path))

    return paths
