from __future__ import annotations

import statistics

import pytest

from crossroute_audit.metrics.stats import (
    benjamini_hochberg,
    bootstrap_ci_median,
    cliffs_delta,
    holm_bonferroni,
    sign_test_pvalue,
)


def test_cliffs_delta_required_values():
    assert cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0
    assert cliffs_delta([4, 5, 6], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_bootstrap_ci_median_constant_and_deterministic():
    constant = bootstrap_ci_median([5.0] * 50, seed=2)
    values = [1.0, 2.0, 4.0, 8.0, 16.0]
    first = bootstrap_ci_median(values, n_boot=500, seed=7)
    repeated = bootstrap_ci_median(values, n_boot=500, seed=7)

    assert constant == (5.0, 5.0)
    assert first == repeated
    assert first[0] <= statistics.median(values) <= first[1]


def test_sign_test_pvalue_is_exact_and_two_sided():
    mixed = sign_test_pvalue([1, 1, 1, 1, -1])
    all_positive = sign_test_pvalue([1] * 10)

    assert mixed == pytest.approx(0.375)
    assert 0 < mixed <= 1
    assert all_positive == pytest.approx(2 / 2**10)
    assert all_positive < mixed


def test_sign_test_drops_ties():
    assert sign_test_pvalue([0.0, 0.0, 1.0, -1.0]) == 1.0
    assert sign_test_pvalue([0.0, 0.0]) == 1.0


def test_holm_bonferroni_required_values_and_original_order():
    assert holm_bonferroni([0.01, 0.04, 0.03], 0.05) == [
        True,
        False,
        False,
    ]
    assert holm_bonferroni([0.06, 0.001, 0.02], 0.05) == [
        False,
        True,
        True,
    ]


def test_benjamini_hochberg_required_extremes_and_original_order():
    assert benjamini_hochberg([0.001, 0.002, 0.003]) == [True, True, True]
    assert benjamini_hochberg([0.9, 0.8, 1.0]) == [False, False, False]
    assert benjamini_hochberg([0.2, 0.001, 0.02], alpha=0.05) == [
        False,
        True,
        True,
    ]


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (cliffs_delta, ([], [1.0])),
        (cliffs_delta, ([1.0], [])),
        (bootstrap_ci_median, ([],)),
        (sign_test_pvalue, ([],)),
        (holm_bonferroni, ([],)),
        (benjamini_hochberg, ([],)),
    ],
)
def test_empty_inputs_raise_clear_value_error(function, args):
    with pytest.raises(ValueError, match="empty"):
        function(*args)


def test_invalid_statistical_parameters_raise_value_error():
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci_median([1.0], n_boot=0)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci_median([1.0], alpha=1.0)
    with pytest.raises(ValueError, match="within"):
        holm_bonferroni([1.1])
    with pytest.raises(ValueError, match="finite"):
        benjamini_hochberg([float("nan")])
