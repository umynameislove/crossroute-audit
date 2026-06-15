from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossroute_audit.io.audit_report import audit_report_for_manifest


def sample(sample_id: str, **overrides) -> dict:
    base = {
        "sample_id": sample_id,
        "image_path": f"{sample_id}.jpg",
        "question": "Is the target present?",
        "target_answer": "yes",
        "target_token_policy": "first_generated_token",
        "expected_visual_dependency": "high",
        "text_only_answerable": "no",
        "counterfactual_image_path": None,
        "control_type": "clean",
        "expected_flip": None,
        "label": None,
        "notes": "Mock sample for audit-report batch tests.",
    }
    base.update(overrides)
    return base


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def write_manifest(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )


def write_artifacts(
    root: Path,
    sample_id: str,
    control_status: dict,
    causal_by_layer: dict,
    attribution_mass: dict,
) -> None:
    write_json(root / "control" / f"control_status_{sample_id}.json", control_status)
    write_json(
        root / "causal" / f"causal_effect_{sample_id}.json",
        {
            "sample_id": sample_id,
            "target_answer": "yes",
            "C_by_layer": causal_by_layer,
            "stability": {},
            "settings": {},
        },
    )
    write_json(
        root / "attribution" / f"attribution_mass_{sample_id}.json",
        {
            "sample_id": sample_id,
            "target_answer": "yes",
            "attribution_mass": attribution_mass,
            "settings": {},
        },
    )


def clean_control_status(sample_id: str, text_only: str = "no") -> dict:
    return {
        "sample_id": sample_id,
        "target_answer": "yes",
        "which": ["text_only", "negative_control"],
        "controls": {
            "text_only": {"target_logit": 0.1, "answerable": text_only},
            "negative_control": {"effect": 0.0},
        },
    }


def test_audit_report_for_manifest_combines_artifacts(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest_path,
        [
            sample("sample_001"),
            sample("sample_002", text_only_answerable="yes"),
        ],
    )
    causal = {
        "image": {"0": 0.1, "1": 0.2, "2": 0.3},
        "text": {"0": 0.1, "1": 0.2, "2": 0.3},
    }
    attribution = {
        "image": {"0": 1.6, "1": 1.4, "2": 1.2},
        "text": {"0": 0.1, "1": 0.2, "2": 0.3},
    }
    write_artifacts(
        tmp_path,
        "sample_001",
        clean_control_status("sample_001"),
        causal,
        attribution,
    )
    write_artifacts(
        tmp_path,
        "sample_002",
        clean_control_status("sample_002", text_only="yes"),
        causal,
        attribution,
    )

    paths = audit_report_for_manifest(
        manifest_path,
        tmp_path / "control",
        tmp_path / "causal",
        tmp_path / "attribution",
        tmp_path / "out",
    )

    assert [Path(path).name for path in paths] == [
        "audit_report_sample_001.json",
        "audit_report_sample_002.json",
    ]

    first = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    assert first["rank_alignment"] == pytest.approx({"image": -1.0, "text": 1.0})
    assert first["diagnosis"]["diagnosis"] == "false_attribution_persistence"
    assert set(first["secondary"]["attribution_flow_gap"]) == {"image", "text"}
    assert set(first["secondary"]["attribution_flow_gap"]["image"]) == {
        "0",
        "1",
        "2",
    }
    assert first["secondary"]["flow_retention"] == pytest.approx(
        {"0": 1.0, "1": 2.0, "2": 3.0}
    )
    assert first["settings"] == {
        "attr_thresh": 1.0,
        "causal_thresh": 0.5,
        "align_thresh": 0.0,
    }

    second = json.loads(Path(paths[1]).read_text(encoding="utf-8"))
    assert second["diagnosis"]["diagnosis"] == "language_prior"


def test_audit_report_for_manifest_reports_missing_artifact(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(manifest_path, [sample("sample_001")])

    with pytest.raises(FileNotFoundError, match="control_status_sample_001.json"):
        audit_report_for_manifest(
            manifest_path,
            tmp_path / "control",
            tmp_path / "causal",
            tmp_path / "attribution",
            tmp_path / "out",
        )
