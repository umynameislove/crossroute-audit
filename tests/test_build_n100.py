from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossroute_audit.io.schema import validate_artifact
from scripts.build_n100_dataset import (
    build_new_only_records,
    build_n100_records,
    deterministic_sample,
    extract_yes_no_answer,
    make_vqa_record,
    normalize_target_answer,
    repo_relative_path,
    write_manifest_atomic,
)


def _pilot_record(index: int) -> dict:
    return {
        "sample_id": f"pilot_{index:04d}",
        "source": "Openverse/flickr | https://example.test/source",
        "image_path": f"data/images/pilot/pilot_{index:04d}.jpg",
        "question": f"Is pilot object {index} visible?",
        "target_answer": "yes",
        "target_token_policy": "first_generated_token",
        "expected_visual_dependency": "high",
        "text_only_answerable": "no",
        "control_type": "clean",
        "notes": "license=see source page; creator=see source page",
    }


def _new_record(index: int, tmp_path: Path) -> dict:
    return make_vqa_record(
        sequence_id=index,
        image_path=tmp_path / "data" / "images" / "vqa_visual" / f"vqa_{index:04d}.jpg",
        question=f"Is there a new object {index} in the image?",
        target_answer="yes" if index % 2 else "no",
        dataset_name="fake/vqa",
        split="validation",
        source_index=index,
        source_metadata={"question_id": index, "image_id": f"img_{index}"},
        repo_root=tmp_path,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("yes", "yes"),
        (" YES ", "yes"),
        ("No.", "no"),
        ("maybe", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_target_answer_keeps_only_yes_no(raw, expected):
    assert normalize_target_answer(raw) == expected


def test_extract_yes_no_answer_rejects_non_yes_no_and_ties():
    assert extract_yes_no_answer({"answer": "yes"}) == "yes"
    assert extract_yes_no_answer({"answers": [{"answer": "no"}, {"answer": "no"}]}) == "no"
    assert extract_yes_no_answer({"answer": "cat"}) is None
    assert extract_yes_no_answer({"answers": [{"answer": "yes"}, {"answer": "no"}]}) is None


def test_make_vqa_record_builds_relative_schema_valid_path(tmp_path):
    record = _new_record(1, tmp_path)

    assert record["sample_id"] == "vqa_0001"
    assert record["image_path"] == "data/images/vqa_visual/vqa_0001.jpg"
    assert record["target_answer"] == "yes"
    assert record["target_token_policy"] == "exact_token"
    assert record["expected_visual_dependency"] == "high"
    assert record["text_only_answerable"] == "unknown"
    assert record["control_type"] == "clean"
    assert record["source"] == "HuggingFace datasets | fake/vqa | validation"
    assert "question_id=1" in record["notes"]
    validate_artifact(record, "manifest")


def test_make_vqa_record_rejects_invalid_answer(tmp_path):
    with pytest.raises(ValueError, match="yes/no"):
        make_vqa_record(
            sequence_id=1,
            image_path=tmp_path / "data/images/vqa_visual/vqa_0001.jpg",
            question="Is this valid?",
            target_answer="cat",
            dataset_name="fake/vqa",
            split="validation",
            source_index=0,
            repo_root=tmp_path,
        )


def test_repo_relative_path_rejects_paths_outside_repo(tmp_path):
    inside = tmp_path / "data" / "images" / "vqa_visual" / "vqa_0001.jpg"
    assert repo_relative_path(inside, repo_root=tmp_path) == (
        "data/images/vqa_visual/vqa_0001.jpg"
    )

    with pytest.raises(ValueError, match="outside repository root"):
        repo_relative_path(Path("/tmp/outside.jpg"), repo_root=tmp_path)


def test_write_manifest_atomic_refuses_to_overwrite_samples_jsonl(tmp_path):
    pilot_manifest = tmp_path / "data" / "manifest" / "samples.jsonl"
    pilot_manifest.parent.mkdir(parents=True)

    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_manifest_atomic([], pilot_manifest, pilot_manifest=pilot_manifest)


def test_write_manifest_atomic_writes_schema_valid_jsonl(tmp_path):
    records = [_pilot_record(1)]
    output_manifest = tmp_path / "data" / "manifest" / "samples_n100.jsonl"
    pilot_manifest = tmp_path / "data" / "manifest" / "samples.jsonl"

    write_manifest_atomic(records, output_manifest, pilot_manifest=pilot_manifest)

    lines = output_manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["sample_id"] == "pilot_0001"
    validate_artifact(loaded, "manifest")


def test_deterministic_sample_is_seeded_and_does_not_mutate_input():
    records = [{"sample_id": str(index)} for index in range(10)]
    original = [dict(record) for record in records]

    first = deterministic_sample(records, 5, seed=7)
    repeated = deterministic_sample(records, 5, seed=7)
    different = deterministic_sample(records, 5, seed=8)

    assert first == repeated
    assert first != different
    assert records == original


def test_build_n100_records_uses_fake_pilot_and_new_data(tmp_path):
    pilot_records = [_pilot_record(index) for index in range(1, 31)]
    new_records = [_new_record(index, tmp_path) for index in range(1, 76)]

    manifest = build_n100_records(pilot_records, new_records, total=100, seed=3)
    repeated = build_n100_records(pilot_records, new_records, total=100, seed=3)

    assert manifest == repeated
    assert len(manifest) == 100
    assert sum(record["sample_id"].startswith("pilot_") for record in manifest) == 30
    assert sum(record["sample_id"].startswith("vqa_") for record in manifest) == 70
    assert len({record["sample_id"] for record in manifest}) == 100
    assert len({record["image_path"] for record in manifest}) == 100
    assert all(not Path(record["image_path"]).is_absolute() for record in manifest)
    assert all(record["target_answer"] in {"yes", "no"} for record in manifest)
    for record in manifest:
        validate_artifact(record, "manifest")


def test_build_n100_records_preserves_schema_source_fields(tmp_path):
    pilot_records = [_pilot_record(index) for index in range(1, 31)]
    new_records = [_new_record(index, tmp_path) for index in range(1, 71)]

    manifest = build_n100_records(pilot_records, new_records, total=100, seed=0)

    pilot = next(record for record in manifest if record["sample_id"] == "pilot_0001")
    assert pilot["source"].startswith("Openverse/flickr")
    assert "license=see source page" in pilot["notes"]

    new = next(record for record in manifest if record["sample_id"] == "vqa_0001")
    assert new["source"] == "HuggingFace datasets | fake/vqa | validation"
    assert "image_id=img_1" in new["notes"]


def test_build_new_only_records_keeps_sorted_vqa_visual_records(tmp_path):
    new_records = [_new_record(index, tmp_path) for index in range(3, 0, -1)]

    manifest = build_new_only_records(new_records, total=3)

    assert [record["sample_id"] for record in manifest] == [
        "vqa_0001",
        "vqa_0002",
        "vqa_0003",
    ]
    assert [record["image_path"] for record in manifest] == [
        "data/images/vqa_visual/vqa_0001.jpg",
        "data/images/vqa_visual/vqa_0002.jpg",
        "data/images/vqa_visual/vqa_0003.jpg",
    ]
    assert all(record["question_type"] == "yes_no" for record in manifest)


def test_build_new_only_records_rejects_missing_metadata(tmp_path):
    new_records = [
        _new_record(1, tmp_path),
        _new_record(3, tmp_path),
        _new_record(4, tmp_path),
    ]

    with pytest.raises(ValueError, match="missing metadata"):
        build_new_only_records(new_records, total=3)
