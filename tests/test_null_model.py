import pytest

from crossroute_audit.metrics.null_model import (
    _aligned_vals,
    _spearman,
    aggregate_null,
    null_pvalue,
    null_zscore,
    sample_null_report,
)


ALIGNED_ATTR = {str(i): float(i) for i in range(12)}
ALIGNED_CAUSAL = {str(i): float(i) for i in range(12)}
RANDOM_ATTR = {
    "0": 3.0,
    "1": 9.0,
    "2": 1.0,
    "3": 7.0,
    "4": 2.0,
    "5": 11.0,
    "6": 0.0,
    "7": 5.0,
    "8": 10.0,
    "9": 4.0,
    "10": 8.0,
    "11": 6.0,
}


def test_aligned_vals_sorts_layers_by_integer_value():
    attr = {"10": 10.0, "2": 2.0, "1": 1.0}
    causal = {"10": 100.0, "2": 20.0, "1": 10.0}

    assert _aligned_vals(attr, causal) == ([1.0, 2.0, 10.0], [10.0, 20.0, 100.0])


def test_aligned_vals_requires_matching_layers():
    with pytest.raises(ValueError, match="same layers"):
        _aligned_vals({"0": 1.0}, {"1": 1.0})


def test_aligned_vals_rejects_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        _aligned_vals({}, {})


def test_aligned_vals_rejects_non_integer_layer_key():
    with pytest.raises(ValueError, match="integer-like strings"):
        _aligned_vals({"layer_0": 1.0}, {"layer_0": 1.0})


def test_null_zscore_is_deterministic_for_seed():
    first = null_zscore(ALIGNED_ATTR, ALIGNED_CAUSAL, n_perm=500, seed=0)
    second = null_zscore(ALIGNED_ATTR, ALIGNED_CAUSAL, n_perm=500, seed=0)

    assert first == second


def test_null_pvalue_is_deterministic_for_seed():
    first = null_pvalue(ALIGNED_ATTR, ALIGNED_CAUSAL, n_perm=500, seed=0)
    second = null_pvalue(ALIGNED_ATTR, ALIGNED_CAUSAL, n_perm=500, seed=0)

    assert first == second


def test_aligned_sample_is_significant_against_null():
    assert null_zscore(ALIGNED_ATTR, ALIGNED_CAUSAL, n_perm=500, seed=0) >= 2.0
    assert null_pvalue(ALIGNED_ATTR, ALIGNED_CAUSAL, n_perm=500, seed=0) <= 0.05


def test_aligned_zscore_is_larger_than_random_zscore():
    z_aligned = null_zscore(ALIGNED_ATTR, ALIGNED_CAUSAL, n_perm=500, seed=0)
    z_random = null_zscore(RANDOM_ATTR, ALIGNED_CAUSAL, n_perm=500, seed=0)

    assert z_aligned > z_random


def test_null_zscore_rejects_invalid_n_perm():
    with pytest.raises(ValueError, match="positive integer"):
        null_zscore(ALIGNED_ATTR, ALIGNED_CAUSAL, n_perm=0)


def test_null_pvalue_rejects_invalid_n_perm():
    with pytest.raises(ValueError, match="positive integer"):
        null_pvalue(ALIGNED_ATTR, ALIGNED_CAUSAL, n_perm=0)


def test_constant_input_falls_back_to_zero_spearman():
    assert _spearman([1.0, 1.0, 1.0], [0.0, 1.0, 2.0]) == 0.0
    report = sample_null_report(
        {"0": 1.0, "1": 1.0, "2": 1.0},
        {"0": 0.0, "1": 1.0, "2": 2.0},
        n_perm=50,
        seed=0,
    )
    assert report["spearman"] == 0.0
    assert report["null_zscore"] == 0.0


def test_sample_null_report_has_required_keys():
    report = sample_null_report(
        ALIGNED_ATTR,
        ALIGNED_CAUSAL,
        n_perm=300,
        seed=7,
    )

    assert set(report) == {
        "spearman",
        "null_zscore",
        "null_pvalue",
        "n_perm",
        "seed",
    }
    assert report["n_perm"] == 300
    assert report["seed"] == 7


def test_aggregate_null_is_deterministic_for_seed():
    attr_by_sample = {"aligned": ALIGNED_ATTR, "random": RANDOM_ATTR}
    causal_by_sample = {"aligned": ALIGNED_CAUSAL, "random": ALIGNED_CAUSAL}

    first = aggregate_null(attr_by_sample, causal_by_sample, n_perm=300, seed=3)
    second = aggregate_null(attr_by_sample, causal_by_sample, n_perm=300, seed=3)

    assert first == second


def test_aggregate_null_has_required_keys_and_sorted_samples():
    attr_by_sample = {"b_random": RANDOM_ATTR, "a_aligned": ALIGNED_ATTR}
    causal_by_sample = {"b_random": ALIGNED_CAUSAL, "a_aligned": ALIGNED_CAUSAL}

    report = aggregate_null(attr_by_sample, causal_by_sample, n_perm=300, seed=3)

    assert set(report) == {
        "mean_zscore",
        "median_zscore",
        "significant_fraction",
        "n_samples",
        "alpha",
        "samples",
    }
    assert report["n_samples"] == 2
    assert report["alpha"] == 0.05
    assert list(report["samples"]) == ["a_aligned", "b_random"]
    assert 0.0 <= report["significant_fraction"] <= 1.0


def test_aggregate_null_rejects_sample_id_mismatch():
    with pytest.raises(ValueError, match="same samples"):
        aggregate_null({"a": ALIGNED_ATTR}, {"b": ALIGNED_CAUSAL})


def test_aggregate_null_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="between 0 and 1"):
        aggregate_null({"a": ALIGNED_ATTR}, {"a": ALIGNED_CAUSAL}, alpha=1.0)


def test_inputs_are_not_mutated():
    attr = dict(ALIGNED_ATTR)
    causal = dict(ALIGNED_CAUSAL)
    attr_by_sample = {"s": dict(attr)}
    causal_by_sample = {"s": dict(causal)}

    sample_null_report(attr, causal, n_perm=100, seed=0)
    aggregate_null(attr_by_sample, causal_by_sample, n_perm=100, seed=0)

    assert attr == ALIGNED_ATTR
    assert causal == ALIGNED_CAUSAL
    assert attr_by_sample == {"s": ALIGNED_ATTR}
    assert causal_by_sample == {"s": ALIGNED_CAUSAL}
