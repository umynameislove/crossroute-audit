from __future__ import annotations

import csv

import pytest

from crossroute_audit.synthetic.run_synthetic import (
    FAULTS,
    _make_case,
    run_synthetic_suite,
)


def test_make_case_rejects_unknown_fault():
    with pytest.raises(ValueError, match="unknown_fault"):
        _make_case("unknown_fault")


def test_run_synthetic_suite_detects_all_faults_and_writes_csv(tmp_path):
    out_csv = tmp_path / "validation" / "benchmark_summary.csv"

    result = run_synthetic_suite(out_csv)

    assert result["total"] == len(FAULTS)
    assert result["detected"] == len(FAULTS)
    assert all(row["detected"] is True for row in result["rows"])
    assert [row["fault"] for row in result["rows"]] == FAULTS

    with out_csv.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert [row["fault"] for row in rows] == FAULTS
    assert all(row["expected"] == row["got"] for row in rows)
    assert all(row["detected"] == "True" for row in rows)
