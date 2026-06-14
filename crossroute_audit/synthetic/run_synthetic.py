"""Synthetic fault suite for validating the diagnosis detector."""
from __future__ import annotations

import csv
from pathlib import Path

from crossroute_audit.metrics.diagnosis import diagnose


FAULTS = [
    "false_attribution",
    "language_prior",
    "modality_drop",
    "negative_control",
]


def _make_case(kind):
    """Return a synthetic detector case and its expected diagnosis."""
    clean_ctrl = {
        "controls": {
            "text_only": {"answerable": "no"},
            "negative_control": {"effect": 0.0},
        }
    }
    if kind == "false_attribution":
        return (
            clean_ctrl,
            {"image": {0: 0.1, 1: 0.2, 2: 0.3}},
            {"image": {0: 1.5, 1: 1.6, 2: 1.7}},
            {"image": -0.8},
            "false_attribution_persistence",
        )
    if kind == "language_prior":
        control_status = {
            "controls": {
                "text_only": {"answerable": "yes"},
                "negative_control": {"effect": 0.0},
            }
        }
        return (
            control_status,
            {"image": {0: 0.1, 1: 0.2}},
            {"image": {0: 1.5, 1: 1.6}},
            {"image": -0.8},
            "language_prior",
        )
    if kind == "modality_drop":
        return (
            clean_ctrl,
            {"image": {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}},
            {"image": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2}},
            {"image": 0.5},
            "modality_drop",
        )
    if kind == "negative_control":
        control_status = {
            "controls": {
                "text_only": {"answerable": "no"},
                "negative_control": {"effect": 0.2},
            }
        }
        return (
            control_status,
            {"image": {0: 0.8, 1: 0.9, 2: 1.0}},
            {"image": {0: 0.2, 1: 0.3, 2: 0.4}},
            {"image": 0.8},
            "low_confidence",
        )
    raise ValueError(kind)


def run_synthetic_suite(out_csv) -> dict:
    """Run all synthetic faults and write a benchmark-summary CSV."""
    rows = []
    detected = 0
    for kind in FAULTS:
        control_status, causal, attribution, rank_alignment, expected = _make_case(kind)
        got = diagnose(
            control_status,
            causal,
            attribution,
            rank_alignment,
        )["diagnosis"]
        ok = got == expected
        detected += int(ok)
        rows.append(
            {
                "fault": kind,
                "expected": expected,
                "got": got,
                "detected": ok,
            }
        )

    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(
            out_file,
            fieldnames=["fault", "expected", "got", "detected"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {"total": len(FAULTS), "detected": detected, "rows": rows}
