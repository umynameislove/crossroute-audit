import copy

import pytest

from crossroute_audit.metrics.robustness import (
    _layer_profile,
    _validate_profile_pair,
    bootstrap_stability,
    detrend_sensitivity,
    robustness_report,
    topk_curve,
)


ATTR = {"0": 4.0, "1": 3.0, "2": 2.0, "3": 1.0}
CAUSAL = {"0": 4.0, "1": 3.0, "2": 2.0, "3": 1.0}
ATTR_BY_SAMPLE = {
    "s1": {"0": 1.0, "1": 2.0, "2": 3.0, "3": 4.0},
    "s2": {"0": 2.0, "1": 3.0, "2": 4.0, "3": 5.0},
    "s3": {"0": 3.0, "1": 4.0, "2": 5.0, "3": 6.0},
}
CAUSAL_BY_SAMPLE = {
    "s1": {"0": 1.0, "1": 2.0, "2": 3.0, "3": 4.0},
    "s2": {"0": 2.0, "1": 3.0, "2": 4.0, "3": 5.0},
    "s3": {"0": 3.0, "1": 4.0, "2": 5.0, "3": 6.0},
}


def test_topk_curve_full_k_is_one():
    assert topk_curve(ATTR, CAUSAL, ks=[4])[4] == pytest.approx(1.0)


def test_topk_curve_default_returns_all_layer_counts():
    assert set(topk_curve(ATTR, CAUSAL)) == {1, 2, 3, 4}


def test_topk_curve_keeps_requested_k_values():
    assert set(topk_curve(ATTR, CAUSAL, ks=[1, 2])) == {1, 2}


@pytest.mark.parametrize("ks", ([0], [5], []))
def test_topk_curve_rejects_invalid_k_values(ks):
    with pytest.raises(ValueError):
        topk_curve(ATTR, CAUSAL, ks=ks)


def test_topk_curve_rejects_mismatched_layers():
    with pytest.raises(ValueError, match="same layers"):
        topk_curve({"0": 1.0}, {"1": 1.0})


def test_validate_profile_pair_rejects_non_integer_layer_key():
    with pytest.raises(ValueError, match="integer-like strings"):
        _validate_profile_pair({"layer_0": 1.0}, {"layer_0": 1.0})


def test_detrend_sensitivity_symmetric_data_is_insensitive_to_reducer():
    report = detrend_sensitivity(ATTR_BY_SAMPLE, CAUSAL_BY_SAMPLE)

    assert report["absolute_difference"] <= 1e-9


def test_detrend_sensitivity_is_deterministic():
    assert detrend_sensitivity(ATTR_BY_SAMPLE, CAUSAL_BY_SAMPLE) == detrend_sensitivity(
        ATTR_BY_SAMPLE,
        CAUSAL_BY_SAMPLE,
    )


def test_detrend_sensitivity_rejects_sample_mismatch():
    with pytest.raises(ValueError, match="same samples"):
        detrend_sensitivity({"s1": ATTR}, {"s2": CAUSAL})


def test_layer_profile_mean_reducer():
    assert _layer_profile(ATTR_BY_SAMPLE, reducer="mean") == {
        "0": pytest.approx(2.0),
        "1": pytest.approx(3.0),
        "2": pytest.approx(4.0),
        "3": pytest.approx(5.0),
    }


def test_layer_profile_median_reducer():
    assert _layer_profile(ATTR_BY_SAMPLE, reducer="median") == {
        "0": pytest.approx(2.0),
        "1": pytest.approx(3.0),
        "2": pytest.approx(4.0),
        "3": pytest.approx(5.0),
    }


def test_layer_profile_rejects_bad_reducer():
    with pytest.raises(ValueError, match="mean' or 'median"):
        _layer_profile(ATTR_BY_SAMPLE, reducer="bad")


def test_bootstrap_stability_constant_values_have_zero_width_ci():
    report = bootstrap_stability([2.0, 2.0, 2.0, 2.0], seeds=(0, 1, 2), n_boot=100)

    assert report["median"] == pytest.approx(2.0)
    assert all(ci == [2.0, 2.0] for ci in report["ci_by_seed"].values())
    assert all(width == pytest.approx(0.0) for width in report["width_by_seed"].values())


def test_bootstrap_stability_is_deterministic_for_seed_sequence():
    first = bootstrap_stability([1.0, 2.0, 3.0, 4.0], seeds=(0, 1), n_boot=100)
    second = bootstrap_stability([1.0, 2.0, 3.0, 4.0], seeds=(0, 1), n_boot=100)

    assert first == second


def test_bootstrap_stability_rejects_empty_values():
    with pytest.raises(ValueError, match="values must not be empty"):
        bootstrap_stability([])


def test_bootstrap_stability_rejects_empty_seeds():
    with pytest.raises(ValueError, match="seeds must not be empty"):
        bootstrap_stability([1.0, 2.0], seeds=())


def test_bootstrap_stability_rejects_invalid_n_boot():
    with pytest.raises(ValueError, match="positive integer"):
        bootstrap_stability([1.0, 2.0], n_boot=0)


@pytest.mark.parametrize("alpha", (0, 1))
def test_bootstrap_stability_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="between 0 and 1"):
        bootstrap_stability([1.0, 2.0], alpha=alpha)


def test_robustness_report_has_required_keys():
    report = robustness_report(
        ATTR,
        CAUSAL,
        ATTR_BY_SAMPLE,
        CAUSAL_BY_SAMPLE,
        [1.0, 2.0, 3.0],
        seeds=(0, 1),
    )

    assert set(report) == {
        "topk_curve",
        "detrend_sensitivity",
        "bootstrap_stability",
        "stable",
    }
    assert report["stable"] is True


def test_robustness_report_rejects_negative_stable_threshold():
    with pytest.raises(ValueError, match="non-negative"):
        robustness_report(
            ATTR,
            CAUSAL,
            ATTR_BY_SAMPLE,
            CAUSAL_BY_SAMPLE,
            [1.0, 2.0],
            stable_threshold=-1,
        )


def test_inputs_are_not_mutated():
    attr = copy.deepcopy(ATTR)
    causal = copy.deepcopy(CAUSAL)
    attr_by_sample = copy.deepcopy(ATTR_BY_SAMPLE)
    causal_by_sample = copy.deepcopy(CAUSAL_BY_SAMPLE)

    topk_curve(attr, causal)
    detrend_sensitivity(attr_by_sample, causal_by_sample)
    bootstrap_stability([1.0, 2.0, 3.0], seeds=(0, 1), n_boot=100)
    robustness_report(
        attr,
        causal,
        attr_by_sample,
        causal_by_sample,
        [1.0, 2.0, 3.0],
        seeds=(0, 1),
    )

    assert attr == ATTR
    assert causal == CAUSAL
    assert attr_by_sample == ATTR_BY_SAMPLE
    assert causal_by_sample == CAUSAL_BY_SAMPLE
