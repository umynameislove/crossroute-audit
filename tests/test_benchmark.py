from __future__ import annotations

import csv
import random

import pytest

from crossroute_audit.metrics.diagnosis import diagnose
from crossroute_audit.synthetic.benchmark import (
    FAULT_CLASSES,
    generate_case,
    run_benchmark,
)


def test_every_class_is_detected_across_random_seeds():
    rng = random.Random(123)
    for kind in FAULT_CLASSES:
        for case_index in range(50):
            ctrl, causal, attr, rank, routing, expected = generate_case(kind, rng)
            got = diagnose(ctrl, causal, attr, rank, routing_proxy=routing)["diagnosis"]
            assert got == expected, (
                f"{kind} case #{case_index} crossed a detector boundary: "
                f"expected={expected!r}, got={got!r}, "
                f"control_status={ctrl!r}, causal={causal!r}, "
                f"attribution={attr!r}, rank_alignment={rank!r}, "
                f"routing_proxy={routing!r}"
            )


def test_run_benchmark_writes_csv_and_reports_accuracy(tmp_path):
    out = tmp_path / "bench" / "benchmark.csv"
    summary = run_benchmark(out, n_per_fault=30, seed=0)

    assert out.is_file(), f"benchmark CSV was not written to {out}"
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    bad_rows = [row for row in rows if row["correct"] != "True"]
    expected_rows = 30 * len(FAULT_CLASSES)
    assert len(rows) == expected_rows, (
        f"unexpected benchmark row count: expected {expected_rows}, got {len(rows)}"
    )
    assert summary["total"] == expected_rows, (
        f"summary total disagrees with CSV rows: {summary['total']} vs {expected_rows}"
    )
    assert summary["accuracy"] == 1.0, (
        f"synthetic benchmark should be perfectly separated; "
        f"accuracy={summary['accuracy']}, bad_rows={bad_rows[:5]}"
    )
    observed_labels = {row["got"] for row in rows} | {row["expected"] for row in rows}
    assert set(summary["per_label"]) <= observed_labels, (
        f"per_label contains labels not present in CSV: "
        f"per_label={summary['per_label']}, observed={observed_labels}"
    )
    expected_labels = {
        "false_attribution_persistence",
        "language_prior",
        "modality_drop",
        "low_confidence",
        "route_break",
        "no_flag",
    }
    assert set(summary["per_label"]) == expected_labels, (
        f"benchmark should cover exactly the detector labels in Đợt 2: "
        f"expected={expected_labels}, got={set(summary['per_label'])}"
    )
    assert set(summary["per_label"]) <= {
        row["got"] for row in rows
    } | {row["expected"] for row in rows}
    for label, metrics in summary["per_label"].items():
        assert metrics["precision"] == 1.0, (
            f"{label} precision should be 1.0; metrics={metrics}, "
            f"confusion={summary['confusion']}"
        )
        assert metrics["recall"] == 1.0, (
            f"{label} recall should be 1.0; metrics={metrics}, "
            f"confusion={summary['confusion']}"
        )
        assert metrics["support"] == 30, (
            f"{label} support should equal n_per_fault=30; metrics={metrics}"
        )
    for expected, got_counts in summary["confusion"].items():
        assert got_counts.get(expected, 0) > 0, (
            f"missing diagonal confusion count for {expected}: {got_counts}"
        )
        off_diagonal = {
            got: count for got, count in got_counts.items() if got != expected
        }
        assert not off_diagonal, (
            f"confusion matrix has off-diagonal mistakes for {expected}: "
            f"{off_diagonal}"
        )


def test_run_benchmark_rejects_non_positive_case_count(tmp_path):
    out = tmp_path / "benchmark.csv"

    with pytest.raises(ValueError, match="n_per_fault must be >= 1"):
        run_benchmark(out, n_per_fault=0)

    assert not out.exists(), (
        f"benchmark should fail before writing a CSV for invalid n_per_fault: {out}"
    )
