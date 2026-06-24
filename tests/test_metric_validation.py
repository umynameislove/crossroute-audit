import random

import pytest

from crossroute_audit.metrics.metric_validation import (
    make_pair,
    recovery_report,
)


def test_make_pair_is_deterministic_for_same_seed():
    left = make_pair("faithful", 24, random.Random(0))
    right = make_pair("faithful", 24, random.Random(0))

    assert left == right


def test_make_pair_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        make_pair("unknown", 24, random.Random(0))


def test_make_pair_requires_at_least_two_layers():
    with pytest.raises(ValueError, match="n_layers must be at least 2"):
        make_pair("faithful", 1, random.Random(0))


def test_make_pair_rejects_negative_noise_level():
    with pytest.raises(ValueError, match="level must be non-negative"):
        make_pair("noisy", 24, random.Random(0), level=-1)


def test_recovery_report_is_deterministic_for_same_seed():
    assert recovery_report(n_layers=24, seed=0) == recovery_report(
        n_layers=24,
        seed=0,
    )


def test_recovery_report_has_required_top_level_keys():
    report = recovery_report(n_layers=24, seed=0)

    assert set(report) == {"faithful", "anti", "shuffled", "noisy_by_level"}


def test_metric_bundles_have_required_metric_keys():
    report = recovery_report(n_layers=24, seed=0)
    expected = {"rank_alignment", "topk_overlap", "profile_emd"}

    assert set(report["faithful"]) == expected
    assert set(report["anti"]) == expected
    assert set(report["shuffled"]) == expected
    for bundle in report["noisy_by_level"].values():
        assert set(bundle) == expected


def test_faithful_pair_recovers_high_alignment_and_low_distribution_shift():
    faithful = recovery_report(n_layers=24, seed=0)["faithful"]

    assert faithful["rank_alignment"] >= 0.95
    assert faithful["topk_overlap"] == pytest.approx(1.0)
    assert faithful["profile_emd"] <= 0.05


def test_anti_pair_recovers_negative_alignment():
    anti = recovery_report(n_layers=24, seed=0)["anti"]

    assert anti["rank_alignment"] <= -0.95


def test_shuffled_pair_has_weak_alignment_for_seeded_case():
    shuffled = recovery_report(n_layers=24, seed=0)["shuffled"]

    assert abs(shuffled["rank_alignment"]) <= 0.5


def test_noisy_rank_alignment_degrades_monotonically_with_noise_level():
    noisy = recovery_report(
        n_layers=24,
        seed=0,
        levels=(0.5, 3.0, 10.0),
    )["noisy_by_level"]

    assert noisy[0.5]["rank_alignment"] > noisy[3.0]["rank_alignment"]
    assert noisy[3.0]["rank_alignment"] > noisy[10.0]["rank_alignment"]
    assert noisy[0.5]["rank_alignment"] == pytest.approx(0.9878, abs=0.02)
    assert noisy[3.0]["rank_alignment"] == pytest.approx(0.7609, abs=0.12)
    assert noisy[10.0]["rank_alignment"] == pytest.approx(0.3635, abs=0.12)


def test_recovery_report_rejects_empty_levels():
    with pytest.raises(ValueError, match="levels must not be empty"):
        recovery_report(n_layers=24, seed=0, levels=())
