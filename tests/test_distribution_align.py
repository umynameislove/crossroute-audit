import pytest

from crossroute_audit.metrics.distribution_align import (
    center_of_mass_shift,
    normalized_emd,
    profile_emd,
)


ALL_L0 = {"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0}
ALL_L1 = {"0": 0.0, "1": 1.0, "2": 0.0, "3": 0.0}
ALL_L3 = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0}


def test_profile_emd_extreme_shift_is_total_cdf_distance():
    assert profile_emd(ALL_L0, ALL_L3) == pytest.approx(3.0)


def test_profile_emd_identical_profiles_is_zero():
    profile = {"0": 0.25, "1": 1.5, "2": 0.25, "3": 0.0}

    assert profile_emd(profile, profile) == pytest.approx(0.0)


def test_profile_emd_one_layer_shift_is_one():
    assert profile_emd(ALL_L0, ALL_L1) == pytest.approx(1.0)


def test_center_of_mass_shift_uses_sorted_layer_positions():
    assert center_of_mass_shift(ALL_L0, ALL_L3) == pytest.approx(-3.0)


def test_normalized_emd_scales_by_layer_count_minus_one():
    assert normalized_emd(ALL_L0, ALL_L3) == pytest.approx(1.0)
    assert normalized_emd(ALL_L0, ALL_L1) == pytest.approx(1 / 3)


def test_different_layer_sets_raise_value_error():
    with pytest.raises(ValueError, match="same layers"):
        profile_emd({"0": 1.0, "1": 0.0}, {"0": 1.0, "2": 0.0})


def test_empty_profile_raises_value_error():
    with pytest.raises(ValueError, match="must not be empty"):
        profile_emd({}, {"0": 1.0})


def test_all_zero_or_all_negative_after_clamp_raises_value_error():
    with pytest.raises(ValueError, match="positive mass"):
        profile_emd({"0": 0.0, "1": 0.0}, {"0": 1.0, "1": 0.0})

    with pytest.raises(ValueError, match="positive mass"):
        profile_emd({"0": -1.0, "1": -2.0}, {"0": 1.0, "1": 0.0})


def test_non_integer_layer_key_raises_value_error():
    with pytest.raises(ValueError, match="parseable as integers"):
        profile_emd({"layer_0": 1.0}, {"0": 1.0})


def test_normalized_emd_requires_at_least_two_layers():
    with pytest.raises(ValueError, match="at least two layers"):
        normalized_emd({"0": 1.0}, {"0": 1.0})


def test_inputs_are_not_mutated():
    attr = {"0": -4.0, "1": 2.0, "2": 0.0, "3": 0.0}
    causal = {"0": 0.0, "1": 1.0, "2": 0.0, "3": 0.0}
    attr_before = attr.copy()
    causal_before = causal.copy()

    assert profile_emd(attr, causal) == pytest.approx(0.0)
    assert attr == attr_before
    assert causal == causal_before


def test_negative_mass_is_clamped_before_normalization():
    attr = {"0": -10.0, "1": 2.0, "2": 0.0, "3": 0.0}

    assert profile_emd(attr, ALL_L1) == pytest.approx(0.0)
