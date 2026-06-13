"""Build and write the per-sample audit report (final GĐ4 artifact)."""
from __future__ import annotations

import json
from pathlib import Path


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
