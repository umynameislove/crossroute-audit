"""Randomized synthetic benchmark for the diagnosis detector."""
from __future__ import annotations

import csv
import random
from pathlib import Path

from crossroute_audit.metrics.diagnosis import diagnose


FAULT_CLASSES = [
    "false_attribution",
    "language_prior",
    "modality_drop",
    "negative_control",
    "route_break",
    "clean",
]


def _img(rng, low, high, n):
    """Return one image group with random layer values in [low, high]."""
    return {"image": {i: round(rng.uniform(low, high), 4) for i in range(n)}}


def generate_case(kind, rng):
    """Generate one safely labeled case for diagnose().

    Returns:
        (control_status, causal_effect, attribution, rank_alignment,
        routing_proxy, expected_diagnosis)
    """
    clean_ctrl = {
        "controls": {
            "text_only": {"answerable": "no"},
            "negative_control": {"effect": round(rng.uniform(-0.03, 0.03), 4)},
        }
    }
    n = rng.choice([3, 4, 5])

    if kind == "false_attribution":
        return (
            clean_ctrl,
            _img(rng, 0.0, 0.4, n),
            _img(rng, 1.2, 2.5, n),
            {"image": round(rng.uniform(-0.95, -0.2), 4)},
            None,
            "false_attribution_persistence",
        )

    if kind == "language_prior":
        ctrl = {
            "controls": {
                "text_only": {"answerable": "yes"},
                "negative_control": {"effect": 0.0},
            }
        }
        return (
            ctrl,
            _img(rng, 0.0, 2.0, n),
            _img(rng, 0.0, 2.0, n),
            {"image": round(rng.uniform(-1.0, 1.0), 4)},
            None,
            "language_prior",
        )

    if kind == "modality_drop":
        return (
            clean_ctrl,
            _img(rng, 0.0, 0.4, n),
            _img(rng, 0.0, 2.0, n),
            {"image": round(rng.uniform(0.2, 0.9), 4)},
            None,
            "modality_drop",
        )

    if kind == "negative_control":
        ctrl = {
            "controls": {
                "text_only": {"answerable": "no"},
                "negative_control": {"effect": round(rng.uniform(0.1, 0.5), 4)},
            }
        }
        return (
            ctrl,
            _img(rng, 0.6, 1.2, n),
            _img(rng, 0.0, 2.0, n),
            {"image": round(rng.uniform(0.2, 0.9), 4)},
            None,
            "low_confidence",
        )

    if kind == "route_break":
        k = rng.choice([1, 2])
        causal = {}
        routing = {}
        for i in range(4):
            if i < k:
                causal[i] = round(rng.uniform(0.8, 1.2), 4)
                routing[i] = round(rng.uniform(0.8, 1.2), 4)
            else:
                causal[i] = round(rng.uniform(0.1, 0.3), 4)
                routing[i] = round(rng.uniform(0.1, 0.3), 4)
        return (
            clean_ctrl,
            {"image": causal},
            _img(rng, 0.0, 0.4, 4),
            {"image": round(rng.uniform(0.2, 0.9), 4)},
            {"image": routing},
            "route_break",
        )

    if kind == "clean":
        return (
            clean_ctrl,
            _img(rng, 0.6, 1.2, n),
            _img(rng, 0.0, 2.0, n),
            {"image": round(rng.uniform(0.2, 0.9), 4)},
            None,
            "no_flag",
        )

    raise ValueError(f"unknown fault class: {kind}")


def _per_label_metrics(rows):
    """Compute precision, recall, and support for each expected label."""
    labels = sorted({row["expected"] for row in rows} | {row["got"] for row in rows})
    out = {}
    for label in labels:
        true_positive = sum(
            1
            for row in rows
            if row["expected"] == label and row["got"] == label
        )
        false_positive = sum(
            1
            for row in rows
            if row["expected"] != label and row["got"] == label
        )
        false_negative = sum(
            1
            for row in rows
            if row["expected"] == label and row["got"] != label
        )
        out[label] = {
            "precision": true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else None,
            "recall": true_positive / (true_positive + false_negative)
            if (true_positive + false_negative)
            else None,
            "support": true_positive + false_negative,
        }
    return out


def run_benchmark(
    out_csv,
    n_per_fault=40,
    seed=0,
    attr_thresh=1.0,
    causal_thresh=0.5,
    align_thresh=0.0,
) -> dict:
    """Run randomized synthetic cases, write per-case CSV, and return metrics."""
    if n_per_fault < 1:
        raise ValueError("n_per_fault must be >= 1")

    rng = random.Random(seed)
    rows = []
    confusion = {}
    for kind in FAULT_CLASSES:
        for _ in range(n_per_fault):
            ctrl, causal, attr, rank, routing, expected = generate_case(kind, rng)
            got = diagnose(
                ctrl,
                causal,
                attr,
                rank,
                attr_thresh=attr_thresh,
                causal_thresh=causal_thresh,
                align_thresh=align_thresh,
                routing_proxy=routing,
            )["diagnosis"]
            rows.append(
                {
                    "fault_class": kind,
                    "expected": expected,
                    "got": got,
                    "correct": got == expected,
                }
            )
            confusion.setdefault(expected, {})
            confusion[expected][got] = confusion[expected].get(got, 0) + 1

    accuracy = sum(row["correct"] for row in rows) / len(rows)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(
            out_file,
            fieldnames=["fault_class", "expected", "got", "correct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {
        "total": len(rows),
        "accuracy": accuracy,
        "per_label": _per_label_metrics(rows),
        "confusion": confusion,
        "n_per_fault": n_per_fault,
        "seed": seed,
    }
