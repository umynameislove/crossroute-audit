from __future__ import annotations

import pytest

from crossroute_audit.metrics.fusion import faithfulness_score


@pytest.mark.parametrize(
    ("alignment", "residual", "control_clean"),
    [
        (-1.0, 0.0, True),
        (1.0, 0.0, True),
        (0.0, 1.0, False),
        (None, -2.0, True),
        (0.5, 100.0, False),
    ],
)
def test_score_is_always_within_unit_interval(
    alignment,
    residual,
    control_clean,
):
    result = faithfulness_score(alignment, residual, control_clean)

    assert 0 <= result["score"] <= 1


def test_score_is_monotonic_in_alignment():
    low = faithfulness_score(-0.8, 0.2, True)["score"]
    middle = faithfulness_score(0.0, 0.2, True)["score"]
    high = faithfulness_score(0.8, 0.2, True)["score"]

    assert low < middle < high


def test_score_is_monotonic_in_completeness():
    low_residual = faithfulness_score(0.4, 0.05, True)["score"]
    high_residual = faithfulness_score(0.4, 2.0, True)["score"]

    assert low_residual > high_residual


def test_failed_control_halves_score_and_sets_flag():
    clean = faithfulness_score(0.4, 0.1, True)
    unverified = faithfulness_score(0.4, 0.1, False)

    assert unverified["score"] == pytest.approx(clean["score"] * 0.5)
    assert clean["flag"] == "ok"
    assert unverified["flag"] == "control_unverified"


def test_none_alignment_uses_neutral_term():
    result = faithfulness_score(None, 0.0, True)

    assert result["components"]["align_term"] == 0.5
    assert result["score"] == pytest.approx(0.5**0.5)


def test_required_clean_and_false_attribution_cases():
    clean = faithfulness_score(0.9, 0.05, True)
    false_attribution = faithfulness_score(-0.8, 0.1, True)

    assert clean["score"] > 0.8
    assert false_attribution["score"] < 0.4


def test_negative_residual_is_clamped_to_zero():
    result = faithfulness_score(0.0, -5.0, True)

    assert result["components"]["complete_term"] == 1.0


@pytest.mark.parametrize(
    ("alignment", "residual", "control_clean", "eps", "message"),
    [
        (1.1, 0.0, True, 1e-9, "within"),
        (float("nan"), 0.0, True, 1e-9, "within"),
        (0.0, float("inf"), True, 1e-9, "finite"),
        (0.0, 0.0, 1, 1e-9, "boolean"),
        (0.0, 0.0, True, 0.0, "eps"),
    ],
)
def test_invalid_inputs_raise_value_error(
    alignment,
    residual,
    control_clean,
    eps,
    message,
):
    with pytest.raises(ValueError, match=message):
        faithfulness_score(
            alignment,
            residual,
            control_clean,
            eps=eps,
        )
