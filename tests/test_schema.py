from __future__ import annotations

import pytest

from crossroute_audit.io.schema import load_schema, validate_artifact


def _valid_manifest() -> dict:
    return {
        "sample_id": "s1",
        "image_path": "data/images/s1.jpg",
        "question": "Is the target visible?",
        "target_answer": "yes",
        "target_token_policy": "first_generated_token",
        "expected_visual_dependency": "high",
        "text_only_answerable": "no",
        "counterfactual_image_path": None,
        "control_type": "clean",
        "expected_flip": None,
        "label": None,
        "notes": "schema test fixture",
    }


def _valid_control_status() -> dict:
    return {
        "sample_id": "s1",
        "target_answer": "yes",
        "which": ["text_only", "negative_control"],
        "controls": {
            "text_only": {"target_logit": 1.2, "answerable": "no"},
            "negative_control": {"effect": 0.0},
        },
    }


def _valid_causal_effect() -> dict:
    return {
        "sample_id": "s1",
        "target_answer": "yes",
        "C_by_layer": {"image": {"0": 0.7}},
        "stability": {
            "image": {
                "0": {
                    "mean": 0.7,
                    "std": 0.0,
                    "max_abs_dev": 0.0,
                }
            }
        },
        "settings": {
            "mode": "ablate",
            "layer_count": 1,
            "groups": ["image"],
            "stability_layer": 0,
            "stability_repeats": 1,
        },
    }


def _valid_audit_report() -> dict:
    return {
        "sample_id": "s1",
        "target_answer": "yes",
        "rank_alignment": {"image": 0.5},
        "secondary": {},
        "control_status": {"text_only": {"answerable": "no"}},
        "diagnosis": {
            "diagnosis": "no_flag",
            "confidence": "high",
            "reasons": ["no strong mismatch"],
        },
        "settings": {},
    }


def test_load_schema_known_and_unknown_names():
    for name in ["manifest", "control_status", "causal_effect", "audit_report"]:
        schema = load_schema(name)
        assert schema["type"] == "object", (
            f"{name} schema should validate object artifacts; schema={schema}"
        )

    with pytest.raises(ValueError, match="unknown schema name"):
        load_schema("does_not_exist")


def test_validate_artifact_accepts_valid_known_artifacts():
    valid_artifacts = {
        "manifest": _valid_manifest(),
        "control_status": _valid_control_status(),
        "causal_effect": _valid_causal_effect(),
        "audit_report": _valid_audit_report(),
    }

    for name, artifact in valid_artifacts.items():
        validate_artifact(artifact, name)


def test_validate_artifact_rejects_missing_required_fields():
    with pytest.raises(ValueError) as exc_info:
        validate_artifact({}, "audit_report")

    message = str(exc_info.value)
    assert "audit_report schema validation failed" in message
    assert "$:" in message, f"missing-field error should point at root: {message}"
    assert "sample_id" in message, (
        f"missing required fields should name the absent field: {message}"
    )


def test_validate_artifact_reports_nested_path_for_bad_diagnosis():
    artifact = _valid_audit_report()
    artifact["diagnosis"]["diagnosis"] = "model_wrong"

    with pytest.raises(ValueError) as exc_info:
        validate_artifact(artifact, "audit_report")

    message = str(exc_info.value)
    assert "$.diagnosis.diagnosis" in message, (
        f"nested enum failure should include the bad field path: {message}"
    )
    assert "model_wrong" in message, (
        f"nested enum failure should include the bad value: {message}"
    )


def test_validate_artifact_rejects_extra_manifest_fields():
    artifact = _valid_manifest()
    artifact["surprise"] = "not in schema"

    with pytest.raises(ValueError) as exc_info:
        validate_artifact(artifact, "manifest")

    message = str(exc_info.value)
    assert "$:" in message, (
        f"additionalProperties failure should point at the root object: {message}"
    )
    assert "surprise" in message, (
        f"additionalProperties failure should name the unexpected field: {message}"
    )


def test_validate_artifact_rejects_bad_causal_layer_keys():
    artifact = _valid_causal_effect()
    artifact["C_by_layer"]["image"] = {"layer_0": 0.7}

    with pytest.raises(ValueError) as exc_info:
        validate_artifact(artifact, "causal_effect")

    message = str(exc_info.value)
    assert "$.C_by_layer.image" in message, (
        f"bad layer-key failure should point to the group layer map: {message}"
    )
    assert "layer_0" in message, (
        f"bad layer-key failure should name the invalid layer key: {message}"
    )
