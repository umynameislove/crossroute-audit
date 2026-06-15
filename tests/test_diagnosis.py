from __future__ import annotations

from crossroute_audit.metrics.diagnosis import diagnose


def control_status(text_only: str = "no", negative_effect: float = 0.0) -> dict:
    return {
        "controls": {
            "text_only": {"answerable": text_only, "target_logit": 0.1},
            "negative_control": {"effect": negative_effect},
        }
    }


def test_text_only_answerable_returns_language_prior_before_mismatch_flags():
    result = diagnose(
        control_status(text_only="yes"),
        causal_effect={"image": {0: 0.1, 1: 0.2}},
        attribution={"image": {0: 2.0, 1: 3.0}},
        rank_alignment={"image": -1.0},
    )

    assert result == {
        "diagnosis": "language_prior",
        "confidence": "high",
        "reasons": ["text_only answerable=yes (model answers without image)"],
    }


def test_false_attribution_persistence_requires_clean_gates_and_low_alignment():
    result = diagnose(
        control_status(),
        causal_effect={"image": {0: 0.1, 1: 0.2, 2: 0.3}},
        attribution={"image": {0: 1.2, 1: 1.4, 2: 1.6}},
        rank_alignment={"image": -0.5},
    )

    assert result == {
        "diagnosis": "false_attribution_persistence",
        "confidence": "high",
        "reasons": [
            "high image attribution + low image causal + low rank alignment + gates clean"
        ],
    }


def test_high_residual_lowers_to_low_confidence_when_no_other_flag_applies():
    result = diagnose(
        control_status(),
        causal_effect={"image": {0: 0.8, 1: 0.9, 2: 1.0}},
        attribution={"image": {0: 0.2, 1: 0.3, 2: 0.4}},
        rank_alignment={"image": 0.8},
        residual=0.3,
    )

    assert result == {
        "diagnosis": "low_confidence",
        "confidence": "low",
        "reasons": ["no strong mismatch"],
    }


def test_missing_alignment_lowers_to_low_confidence_when_no_other_flag_applies():
    result = diagnose(
        control_status(),
        causal_effect={"image": {0: 0.8, 1: 0.9, 2: 1.0}},
        attribution={"image": {0: 0.2, 1: 0.3, 2: 0.4}},
        rank_alignment={},
    )

    assert result["diagnosis"] == "low_confidence"
    assert result["confidence"] == "low"


def test_modality_drop_when_image_causal_effect_is_low_across_most_layers():
    result = diagnose(
        control_status(),
        causal_effect={"image": {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.9}},
        attribution={"image": {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}},
        rank_alignment={"image": 0.4},
    )

    assert result == {
        "diagnosis": "modality_drop",
        "confidence": "high",
        "reasons": ["image causal effect low across most layers"],
    }


def test_route_break_when_routing_proxy_and_causal_drop_at_same_layer():
    result = diagnose(
        control_status(),
        causal_effect={"image": {"0": 1.0, "1": 0.9, "2": 0.2, "3": 0.2}},
        attribution={"image": {"0": 0.2, "1": 0.3, "2": 0.4, "3": 0.5}},
        rank_alignment={"image": 0.8},
        routing_proxy={"image": {"0": 0.9, "1": 0.8, "2": 0.2, "3": 0.2}},
    )

    assert result == {
        "diagnosis": "route_break",
        "confidence": "high",
        "reasons": [
            "routing proxy and image causal effect drop sharply near layer 2"
        ],
    }


def test_route_break_requires_proxy_and_causal_to_drop_together():
    result = diagnose(
        control_status(),
        causal_effect={"image": {"0": 1.0, "1": 0.95, "2": 0.9, "3": 0.85}},
        attribution={"image": {"0": 0.2, "1": 0.3, "2": 0.4, "3": 0.5}},
        rank_alignment={"image": 0.8},
        routing_proxy={"image": {"0": 0.9, "1": 0.8, "2": 0.2, "3": 0.2}},
    )

    assert result == {
        "diagnosis": "no_flag",
        "confidence": "high",
        "reasons": ["no strong mismatch"],
    }


def test_negative_control_not_clean_lowers_confidence():
    result = diagnose(
        control_status(negative_effect=0.2),
        causal_effect={"image": {0: 0.8, 1: 0.9, 2: 1.0}},
        attribution={"image": {0: 0.2, 1: 0.3, 2: 0.4}},
        rank_alignment={"image": 0.8},
    )

    assert result == {
        "diagnosis": "low_confidence",
        "confidence": "low",
        "reasons": ["negative_control not clean"],
    }


def test_no_strong_mismatch_returns_no_flag_when_confidence_is_high():
    result = diagnose(
        control_status(),
        causal_effect={"image": {0: 0.8, 1: 0.9, 2: 1.0}},
        attribution={"image": {0: 0.2, 1: 0.3, 2: 0.4}},
        rank_alignment={"image": 0.8},
    )

    assert result == {
        "diagnosis": "no_flag",
        "confidence": "high",
        "reasons": ["no strong mismatch"],
    }
