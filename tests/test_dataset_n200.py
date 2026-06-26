import copy
import json
import random
from pathlib import Path

import pytest

import scripts.build_n200_dataset as builder
from crossroute_audit.io.schema import validate_artifact


def make_row(i: int, answer: str, dep: str = "high", qtype: str = "presence") -> dict:
    question_by_type = {
        "presence": f"Is there an object {i} in the image?",
        "color": f"What color evidence supports this yes/no question {i}?",
        "count": f"How many visual checks support this yes/no question {i}?",
        "spatial": f"Where is the relevant visual evidence for this yes/no question {i}?",
    }
    return {
        "sample_id": f"s{i}",
        "source": "unit-test",
        "image_path": f"images/{i}.jpg",
        "question": question_by_type[qtype],
        "question_type": builder.tag_question_type(question_by_type[qtype]),
        "target_answer": answer,
        "target_token_policy": "first_generated_token",
        "expected_visual_dependency": dep,
        "text_only_answerable": "unknown",
        "counterfactual_image_path": f"images/cf_{i}.jpg",
        "control_type": "clean",
        "expected_flip": None,
        "label": None,
        "notes": "",
    }


def fake_rows(n_yes=130, n_no=130):
    rows = []
    qtypes = ("presence", "color", "count", "spatial")
    for i in range(n_yes):
        rows.append(make_row(i, "yes", "high" if i % 3 else "low", qtypes[i % 4]))
    offset = n_yes
    for i in range(n_no):
        rows.append(
            make_row(
                offset + i,
                "no",
                "high" if i % 4 else "low",
                qtypes[(i + 1) % 4],
            )
        )
    return rows


def seed_rows():
    rows = []
    for i in range(56):
        rows.append(make_row(i, "yes", "high" if i % 10 else "low", "presence"))
    for i in range(44):
        rows.append(make_row(100 + i, "no", "high" if i % 10 else "low", "presence"))
    return rows


def fake_photos(n=100):
    return [
        {
            "id": str(i),
            "author": f"Author {i}",
            "url": f"https://example.com/photo/{i}",
            "download_url": f"https://example.com/download/{i}",
        }
        for i in range(n)
    ]


def fake_download(photo_id, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(f"fake image {photo_id}".encode("utf-8"))


def patch_fake_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(builder, "_load_seed_rows", lambda path: seed_rows())
    monkeypatch.setattr(builder, "_fetch_picsum_metadata", fake_photos)
    monkeypatch.setattr(builder, "_download_picsum_image", fake_download)
    monkeypatch.setattr(builder, "DEFAULT_IMAGE_DIR", tmp_path / "images")


def test_balance_yes_no_selects_total_and_balances_answers_deterministically():
    rows = fake_rows(20, 20)
    first = builder.balance_yes_no(rows, 21, random.Random(0))
    second = builder.balance_yes_no(rows, 21, random.Random(0))
    counts = builder.Counter(builder._answer(row) for row in first)

    assert len(first) == 21
    assert abs(counts["yes"] - counts["no"]) <= 1
    assert first == second


def test_balance_yes_no_rejects_missing_side():
    with pytest.raises(ValueError, match="not enough yes/no"):
        builder.balance_yes_no(fake_rows(10, 0), 10, random.Random(0))


def test_balance_yes_no_rejects_non_positive_total():
    with pytest.raises(ValueError, match="total must be positive"):
        builder.balance_yes_no(fake_rows(10, 10), 0, random.Random(0))


def test_dedupe_rows_removes_duplicate_sample_id():
    rows = [make_row(1, "yes"), make_row(1, "no")]

    assert builder._dedupe_rows(rows) == [rows[0]]


def test_dedupe_rows_removes_duplicate_image_question():
    rows = [make_row(1, "yes"), make_row(2, "no")]
    rows[1]["image_path"] = rows[0]["image_path"]
    rows[1]["question"] = rows[0]["question"]

    assert builder._dedupe_rows(rows) == [rows[0]]


def test_question_type_caps_accepts_balanced_types():
    builder.question_type_caps(
        [
            make_row(1, "yes", qtype="presence"),
            make_row(2, "no", qtype="color"),
            make_row(3, "yes", qtype="count"),
            make_row(4, "no", qtype="spatial"),
        ],
        max_frac=0.6,
    )


def test_question_type_caps_rejects_dominant_type():
    with pytest.raises(ValueError, match="question type cap exceeded"):
        builder.question_type_caps([make_row(i, "yes") for i in range(10)], max_frac=0.6)


@pytest.mark.parametrize("dep", ("high", "low"))
def test_has_both_visual_dependencies_requires_high_and_low(dep):
    with pytest.raises(ValueError, match="both high and low"):
        builder._has_both_visual_dependencies([make_row(i, "yes", dep=dep) for i in range(4)])


def test_manifest_report_has_expected_keys_and_answer_counts():
    rows = [make_row(1, "yes", "high"), make_row(2, "no", "low")]
    report = builder.manifest_report(rows)

    assert set(report) == {"total", "answers", "visual_dependency", "question_types"}
    assert report["answers"] == {"no": 1, "yes": 1}


def test_select_n200_is_deterministic_with_fake_source(monkeypatch, tmp_path):
    patch_fake_sources(monkeypatch, tmp_path)

    assert builder.select_n200(seed=0, total=200) == builder.select_n200(
        seed=0,
        total=200,
    )


def test_select_n200_changes_with_seed_when_source_is_diverse(monkeypatch, tmp_path):
    patch_fake_sources(monkeypatch, tmp_path)

    assert builder.select_n200(seed=0, total=200) != builder.select_n200(
        seed=1,
        total=200,
    )


def test_select_n200_returns_valid_balanced_unique_rows_with_fake_source(monkeypatch, tmp_path):
    patch_fake_sources(monkeypatch, tmp_path)
    rows = builder.select_n200(seed=0, total=200)
    report = builder.manifest_report(rows)

    assert len(rows) == 200
    assert len({row["sample_id"] for row in rows}) == 200
    assert len({row["image_path"] for row in rows}) == 200
    assert len({(row["image_path"], row["question"]) for row in rows}) == 200
    assert report["answers"] == {"no": 100, "yes": 100}
    assert set(report["visual_dependency"]) == {"high", "low"}
    assert max(report["question_types"].values()) / report["total"] <= 0.6


def test_generated_manifest_current_repo_source():
    rows = [
        json.loads(line)
        for line in Path("data/manifest/samples_n200.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(rows) == 200
    assert len({row["image_path"] for row in rows}) == 200


def test_select_n200_rows_validate_against_schema(monkeypatch, tmp_path):
    patch_fake_sources(monkeypatch, tmp_path)
    for row in builder.select_n200(seed=0, total=200):
        validate_artifact(row, "manifest")


def test_write_jsonl_is_deterministic(tmp_path: Path):
    rows = [make_row(1, "yes"), make_row(2, "no", "low")]
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"

    builder.write_jsonl(rows, left)
    builder.write_jsonl(rows, right)

    assert left.read_bytes() == right.read_bytes()
    assert len(left.read_text(encoding="utf-8").splitlines()) == 2
    assert json.loads(left.read_text(encoding="utf-8").splitlines()[0])["sample_id"] == "s1"


def test_inputs_are_not_mutated(monkeypatch, tmp_path):
    rows = fake_rows(20, 20)
    before = copy.deepcopy(rows)
    patch_fake_sources(monkeypatch, tmp_path)

    builder._dedupe_rows(rows)
    builder.balance_yes_no(rows, 20, random.Random(0))
    builder.question_type_caps(rows)
    builder.manifest_report(rows)
    builder.select_n200(seed=0, total=200)

    assert rows == before
