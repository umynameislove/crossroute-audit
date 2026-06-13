from __future__ import annotations

import pytest

from crossroute_audit.metrics.rank_alignment import (
    rank_alignment,
    rank_alignment_by_group,
)


def test_rank_alignment_returns_positive_one_for_same_order():
    attribution = {0: 0.1, 1: 0.3, 2: 0.8}
    causal = {0: 2.0, 1: 4.0, 2: 9.0}

    assert rank_alignment(attribution, causal) == pytest.approx(1.0)


def test_rank_alignment_returns_negative_one_for_reverse_order():
    attribution = {0: 0.1, 1: 0.3, 2: 0.8}
    causal = {0: 9.0, 1: 4.0, 2: 2.0}

    assert rank_alignment(attribution, causal) == pytest.approx(-1.0)


def test_rank_alignment_uses_common_layers_only():
    attribution = {0: 0.1, 1: 0.3, 2: 0.8}
    causal = {1: 4.0, 2: 9.0, 3: -100.0}

    assert rank_alignment(attribution, causal) == pytest.approx(1.0)


def test_rank_alignment_returns_none_for_too_few_common_layers():
    assert rank_alignment({0: 0.1}, {0: 2.0}) is None


def test_rank_alignment_returns_none_when_one_side_is_constant():
    assert rank_alignment({0: 1.0, 1: 1.0, 2: 1.0}, {0: 1.0, 1: 2.0, 2: 3.0}) is None
    assert rank_alignment({0: 1.0, 1: 2.0, 2: 3.0}, {0: 1.0, 1: 1.0, 2: 1.0}) is None


def test_rank_alignment_by_group_returns_group_rhos():
    attribution = {
        "image": {0: 0.1, 1: 0.3, 2: 0.8},
        "text": {0: 0.9, 1: 0.4, 2: 0.2},
        "unused": {0: 1.0, 1: 2.0},
    }
    causal = {
        "image": {0: 2.0, 1: 4.0, 2: 9.0},
        "text": {0: 1.0, 1: 2.0, 2: 3.0},
    }

    assert rank_alignment_by_group(attribution, causal) == pytest.approx(
        {
            "image": 1.0,
            "text": -1.0,
        }
    )
