from __future__ import annotations

import pytest

from crossroute_audit.metrics.structure_align import (
    detrended_rank_alignment,
    topk_overlap,
)


def test_topk_overlap_required_values():
    assert topk_overlap({"0": 5, "1": 4, "2": 1, "3": 0}, {"0": 1, "1": 0, "2": 5, "3": 4}, k=2) == 0.0
    assert topk_overlap({"0": 5, "1": 4, "2": 1, "3": 0}, {"0": 6, "1": 3, "2": 2, "3": 1}, k=2) == 1.0
    assert topk_overlap({"0": 5, "1": 4, "2": 1, "3": 0}, {"0": 6, "1": 1, "2": 5, "3": 0}, k=2) == 0.5


def test_topk_overlap_validation():
    with pytest.raises(ValueError, match="k must be"):
        topk_overlap({"0": 1, "1": 2}, {"0": 1, "1": 2}, k=3)
    with pytest.raises(ValueError, match="share the same layers"):
        topk_overlap({"0": 1}, {"1": 1}, k=1)


def test_detrended_removes_shared_trend_perfect_match():
    trend = {"0": 10, "1": 7, "2": 4, "3": 1}
    a = {"A": {**trend, "1": trend["1"] + 1}, "B": {**trend, "2": trend["2"] + 1}}
    res = detrended_rank_alignment(a, {k: dict(v) for k, v in a.items()})
    assert res["A"] == pytest.approx(1.0)
    assert res["B"] == pytest.approx(1.0)


def test_detrended_constant_deviation_is_none():
    a = {"A": {"0": 1, "1": 1, "2": 1}, "B": {"0": 1, "1": 1, "2": 1}}
    res = detrended_rank_alignment(a, {k: dict(v) for k, v in a.items()})
    assert res["A"] is None and res["B"] is None


def test_detrended_validation():
    with pytest.raises(ValueError, match="same sample ids"):
        detrended_rank_alignment({"A": {"0": 1}}, {"B": {"0": 1}})
