from __future__ import annotations

import pytest

from crossroute_audit.metrics.flow_diagnostics import (
    _zscore,
    attribution_flow_gap,
    flow_retention,
)


def test_zscore_centers_and_scales_values():
    assert _zscore([1.0, 2.0, 3.0]) == pytest.approx(
        [-1.224744871, 0.0, 1.224744871]
    )


def test_zscore_handles_constant_values_without_dividing_by_zero():
    assert _zscore([4.0, 4.0, 4.0]) == pytest.approx([0.0, 0.0, 0.0])


def test_attribution_flow_gap_is_zero_for_identical_layer_trends():
    attr = {0: 1.0, 1: 2.0, 2: 3.0}
    causal = {0: 10.0, 1: 20.0, 2: 30.0}

    assert attribution_flow_gap(attr, causal) == pytest.approx(
        {0: 0.0, 1: 0.0, 2: 0.0}
    )


def test_attribution_flow_gap_uses_common_layers_only():
    attr = {0: -100.0, 1: 1.0, 2: 2.0}
    causal = {1: 10.0, 2: 20.0, 3: 999.0}

    assert attribution_flow_gap(attr, causal) == pytest.approx({1: 0.0, 2: 0.0})


def test_attribution_flow_gap_returns_empty_dict_when_no_layers_overlap():
    assert attribution_flow_gap({0: 1.0}, {1: 1.0}) == {}


def test_flow_retention_normalizes_by_first_layer():
    assert flow_retention({0: 2.0, 1: 3.0, 2: 1.0}) == pytest.approx(
        {0: 1.0, 1: 1.5, 2: 0.5}
    )


def test_flow_retention_uses_lowest_layer_as_first_layer():
    assert flow_retention({3: 9.0, 1: 3.0, 2: 6.0}) == pytest.approx(
        {3: 3.0, 1: 1.0, 2: 2.0}
    )


def test_flow_retention_handles_empty_and_zero_first_layer():
    assert flow_retention({}) == {}
    assert flow_retention({0: 0.0, 1: 2.0}) == pytest.approx({0: 0.0, 1: 2.0})
