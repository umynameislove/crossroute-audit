"""Threshold-sensitivity analysis for the synthetic diagnosis benchmark."""
from __future__ import annotations

import itertools
import math
import tempfile
from pathlib import Path

from crossroute_audit.synthetic.benchmark import run_benchmark


def sweep_thresholds(
    attr_grid: list[float],
    causal_grid: list[float],
    align_grid: list[float],
    *,
    n_per_fault: int = 20,
    seed: int = 0,
) -> list[dict]:
    """Evaluate every Cartesian-product threshold combination."""
    _validate_grid(attr_grid, "attr_grid")
    _validate_grid(causal_grid, "causal_grid")
    _validate_grid(align_grid, "align_grid")
    if not isinstance(n_per_fault, int) or isinstance(n_per_fault, bool):
        raise ValueError("n_per_fault must be a positive integer")
    if n_per_fault < 1:
        raise ValueError("n_per_fault must be a positive integer")

    sweep = []
    with tempfile.TemporaryDirectory(prefix="crossroute-sensitivity-") as tmp_dir:
        out_csv = Path(tmp_dir) / "benchmark.csv"
        for attr_thresh, causal_thresh, align_thresh in itertools.product(
            attr_grid,
            causal_grid,
            align_grid,
        ):
            summary = run_benchmark(
                out_csv,
                n_per_fault=n_per_fault,
                seed=seed,
                attr_thresh=attr_thresh,
                causal_thresh=causal_thresh,
                align_thresh=align_thresh,
            )
            sweep.append(
                {
                    "attr_thresh": attr_thresh,
                    "causal_thresh": causal_thresh,
                    "align_thresh": align_thresh,
                    "accuracy": summary["accuracy"],
                }
            )
    return sweep


def stable_region(
    sweep: list[dict],
    min_accuracy: float = 0.95,
) -> list[dict]:
    """Return threshold combinations meeting the minimum accuracy."""
    if not math.isfinite(min_accuracy) or not 0 <= min_accuracy <= 1:
        raise ValueError("min_accuracy must be finite and within [0, 1]")
    return [
        row
        for row in sweep
        if row["accuracy"] >= min_accuracy
    ]


def _validate_grid(grid: list[float], name: str) -> None:
    if not grid:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) for value in grid):
        raise ValueError(f"{name} values must be finite")
