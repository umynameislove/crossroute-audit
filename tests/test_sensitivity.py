from __future__ import annotations

import pytest

from crossroute_audit.metrics.sensitivity import stable_region, sweep_thresholds


def test_sweep_thresholds_returns_cartesian_product():
    sweep = sweep_thresholds(
        [0.8, 1.2],
        [0.4, 0.6],
        [-0.1, 0.1],
        n_per_fault=5,
        seed=3,
    )

    assert len(sweep) == 8
    assert all(
        set(row)
        == {"attr_thresh", "causal_thresh", "align_thresh", "accuracy"}
        for row in sweep
    )
    assert all(0 <= row["accuracy"] <= 1 for row in sweep)
    assert {
        (row["attr_thresh"], row["causal_thresh"], row["align_thresh"])
        for row in sweep
    } == {
        (attr_thresh, causal_thresh, align_thresh)
        for attr_thresh in [0.8, 1.2]
        for causal_thresh in [0.4, 0.6]
        for align_thresh in [-0.1, 0.1]
    }


def test_sweep_thresholds_is_deterministic():
    args = ([0.8, 1.2], [0.4], [0.0, 0.2])

    first = sweep_thresholds(*args, n_per_fault=6, seed=9)
    repeated = sweep_thresholds(*args, n_per_fault=6, seed=9)

    assert first == repeated


def test_stable_region_filters_without_reordering():
    sweep = [
        {
            "attr_thresh": 0.8,
            "causal_thresh": 0.4,
            "align_thresh": 0.0,
            "accuracy": 0.94,
        },
        {
            "attr_thresh": 1.0,
            "causal_thresh": 0.5,
            "align_thresh": 0.0,
            "accuracy": 0.95,
        },
        {
            "attr_thresh": 1.2,
            "causal_thresh": 0.6,
            "align_thresh": 0.1,
            "accuracy": 1.0,
        },
    ]

    assert stable_region(sweep) == sweep[1:]


@pytest.mark.parametrize(
    ("attr_grid", "causal_grid", "align_grid", "message"),
    [
        ([], [0.5], [0.0], "attr_grid"),
        ([1.0], [], [0.0], "causal_grid"),
        ([1.0], [0.5], [], "align_grid"),
        ([float("nan")], [0.5], [0.0], "attr_grid"),
    ],
)
def test_sweep_thresholds_rejects_invalid_grids(
    attr_grid,
    causal_grid,
    align_grid,
    message,
):
    with pytest.raises(ValueError, match=message):
        sweep_thresholds(attr_grid, causal_grid, align_grid)


def test_invalid_counts_and_accuracy_threshold_raise_value_error():
    with pytest.raises(ValueError, match="n_per_fault"):
        sweep_thresholds([1.0], [0.5], [0.0], n_per_fault=0)
    with pytest.raises(ValueError, match="min_accuracy"):
        stable_region([], min_accuracy=1.1)
