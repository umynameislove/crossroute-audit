from __future__ import annotations

from collections import Counter

import pytest

from crossroute_audit.io.dataset_loader import build_manifest, tag_question_type
from crossroute_audit.io.schema import validate_artifact


def _records(per_type: int = 5) -> list[dict]:
    templates = {
        "yes_no": "Is object {index} visible?",
        "count": "How many objects are in image {index}?",
        "color": "What color is object {index}?",
        "where": "Where is object {index}?",
        "what": "What is object {index}?",
        "other": "Describe image {index}.",
    }
    return [
        {
            "image_path": f"data/images/{question_type}_{index}.jpg",
            "question": template.format(index=index),
            "target_answer": "yes",
        }
        for question_type, template in templates.items()
        for index in range(per_type)
    ]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Is there a dog?", "yes_no"),
        ("How many cats?", "count"),
        ("What color is the bus?", "color"),
        ("Where is the cat?", "where"),
        ("What is shown?", "what"),
        ("Describe it.", "other"),
    ],
)
def test_tag_question_type_required_examples(question, expected):
    assert tag_question_type(question) == expected


def test_build_manifest_is_deterministic_for_seed():
    records = _records()

    first = build_manifest(records, 20, seed=1)
    repeated = build_manifest(records, 20, seed=1)
    different_seed = build_manifest(records, 20, seed=2)

    assert first == repeated
    assert first != different_seed


def test_build_manifest_balances_question_types():
    manifest = build_manifest(_records(), 20, seed=1)
    counts = Counter(record["question_type"] for record in manifest)

    assert set(counts) == {"yes_no", "count", "color", "what", "where", "other"}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_build_manifest_deduplicates_by_image_and_question():
    records = _records(per_type=1)
    duplicate = dict(records[0], target_answer="no")
    records.append(duplicate)

    manifest = build_manifest(records, 6, seed=0)
    keys = [(record["image_path"], record["question"]) for record in manifest]

    assert len(keys) == len(set(keys)) == 6


def test_build_manifest_rejects_n_larger_than_unique_records():
    records = _records(per_type=1)

    with pytest.raises(ValueError, match="only 6 unique records"):
        build_manifest(records, 7)


def test_build_manifest_outputs_schema_valid_records():
    manifest = build_manifest(_records(), 20, seed=3)

    for record in manifest:
        validate_artifact(record, "manifest")
