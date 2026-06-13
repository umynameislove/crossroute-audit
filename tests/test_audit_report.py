from __future__ import annotations

import json

from jsonschema import Draft202012Validator

from crossroute_audit.io.audit_report import build_audit_report, write_audit_report


def sample(**overrides) -> dict:
    base = {
        "sample_id": "sample_001",
        "image_path": "clean.jpg",
        "question": "Is the target present?",
        "target_answer": "yes",
        "target_token_policy": "first_generated_token",
        "expected_visual_dependency": "high",
        "text_only_answerable": "no",
        "counterfactual_image_path": None,
        "control_type": "clean",
        "expected_flip": None,
        "label": None,
        "notes": "Mock sample for audit-report tests.",
    }
    base.update(overrides)
    return base


def load_audit_report_schema() -> dict:
    with open("schemas/audit_report.schema.json", encoding="utf-8") as schema_file:
        return json.load(schema_file)


def test_build_audit_report_returns_expected_fields():
    control_status = {
        "sample_id": "sample_001",
        "target_answer": "yes",
        "controls": {
            "text_only": {"target_logit": 0.1, "answerable": "no"},
            "negative_control": {"effect": 0.0},
        },
    }
    causal_effect = {"image": {"0": 0.2}}
    attribution = {"image": {"0": 1.5}}
    rank_alignment = {"image": -0.5}
    secondary = {"flow_retention": {"0": 1.0}}
    diagnosis = {
        "diagnosis": "false_attribution_persistence",
        "confidence": "high",
        "reasons": ["mock mismatch"],
    }
    settings = {"metric": "rank_alignment"}

    report = build_audit_report(
        sample(),
        control_status,
        causal_effect,
        attribution,
        rank_alignment,
        secondary,
        diagnosis,
        settings,
    )

    assert report == {
        "sample_id": "sample_001",
        "target_answer": "yes",
        "rank_alignment": rank_alignment,
        "secondary": secondary,
        "control_status": control_status["controls"],
        "diagnosis": diagnosis,
        "settings": settings,
    }


def test_write_audit_report_round_trips_and_validates_schema(tmp_path):
    report = build_audit_report(
        sample(),
        {
            "controls": {
                "text_only": {"target_logit": 0.1, "answerable": "no"},
                "negative_control": {"effect": 0.0},
            }
        },
        causal_effect={"image": {"0": 0.2}},
        attribution={"image": {"0": 1.5}},
        rank_alignment={"image": -0.5},
        secondary={"attribution_flow_gap": {"0": 0.0}},
        diagnosis={
            "diagnosis": "no_flag",
            "confidence": "high",
            "reasons": ["no strong mismatch"],
        },
        settings={"ig_steps": 32},
    )
    out_path = tmp_path / "nested" / "audit_report_sample_001.json"

    write_audit_report(report, out_path)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))

    assert loaded == report
    Draft202012Validator(load_audit_report_schema()).validate(loaded)
