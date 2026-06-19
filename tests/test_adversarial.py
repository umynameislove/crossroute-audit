from __future__ import annotations

import random

import pytest

from crossroute_audit.synthetic.adversarial import (
    degradation_curve,
    noisy_case,
    roc_auc,
)
from crossroute_audit.synthetic.benchmark import FAULT_CLASSES, generate_case


def test_roc_auc_required_value_and_ties():
    assert roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert roc_auc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)


@pytest.mark.parametrize("labels", [[1, 1, 1], [0, 0, 0]])
def test_roc_auc_rejects_single_class(labels):
    with pytest.raises(ValueError, match="both classes"):
        roc_auc([0.1, 0.2, 0.3], labels)


def test_noisy_case_sigma_zero_matches_generate_case_exactly():
    for kind in FAULT_CLASSES:
        expected_rng = random.Random(11)
        noisy_rng = random.Random(11)

        assert noisy_case(kind, noisy_rng, 0.0) == generate_case(
            kind,
            expected_rng,
        )
        assert noisy_rng.getstate() == expected_rng.getstate()


def test_noisy_case_adds_noise_deterministically():
    first = noisy_case("false_attribution", random.Random(3), 0.5)
    repeated = noisy_case("false_attribution", random.Random(3), 0.5)
    clean = noisy_case("false_attribution", random.Random(3), 0.0)

    assert first == repeated
    assert first != clean
    assert first[-1] == clean[-1]


def test_degradation_curve_sigma_zero_matches_benchmark_v1():
    curve = degradation_curve([0.0], n_per_fault=30, seed=0)

    assert curve == [{"sigma": 0.0, "accuracy": 1.0}]


def test_degradation_curve_large_noise_does_not_exceed_baseline():
    curve = degradation_curve([0.0, 5.0], n_per_fault=30, seed=4)

    assert curve[1]["accuracy"] <= curve[0]["accuracy"]


def test_degradation_curve_is_deterministic_for_seed():
    first = degradation_curve([0.0, 0.5, 2.0], n_per_fault=10, seed=9)
    repeated = degradation_curve([0.0, 0.5, 2.0], n_per_fault=10, seed=9)

    assert first == repeated


def test_degradation_accuracy_does_not_depend_on_sigma_list_position():
    alone = degradation_curve([0.5], n_per_fault=30, seed=4)[0]
    after_other_sigma = degradation_curve(
        [0.0, 0.5],
        n_per_fault=30,
        seed=4,
    )[1]

    assert alone == after_other_sigma


def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError, match="sigma"):
        noisy_case("clean", random.Random(0), -0.1)
    with pytest.raises(ValueError, match="n_per_fault"):
        degradation_curve([0.0], n_per_fault=0)
    with pytest.raises(ValueError, match="sigmas"):
        degradation_curve([float("nan")])
    with pytest.raises(ValueError, match="equal length"):
        roc_auc([0.1], [0, 1])
    with pytest.raises(ValueError, match="only 0 and 1"):
        roc_auc([0.1, 0.2], [0, 2])
