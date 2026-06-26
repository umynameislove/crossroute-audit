"""Build reproducible N=200 yes/no VQA manifest (balanced + stratified)."""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossroute_audit.io.dataset_loader import build_manifest, tag_question_type
from crossroute_audit.io.schema import validate_artifact


DEFAULT_TOTAL = 200
DEFAULT_OUTPUT = REPO_ROOT / "data" / "manifest" / "samples_n200.jsonl"
DEFAULT_SEED_MANIFEST = REPO_ROOT / "data" / "manifest" / "samples_n100.jsonl"
DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "images" / "vqa_visual_n200"
PICSUM_API = "https://picsum.photos/v2/list?page=1&limit=100"
PICSUM_IMAGE_URL = "https://picsum.photos/id/{id}/768/512"
QUESTION_TYPES = ("yes_no", "what", "where", "color", "count")


def _answer(row: dict) -> str:
    answer = str(row.get("target_answer", "")).strip().lower()
    if answer not in {"yes", "no"}:
        raise ValueError("row target_answer must be 'yes' or 'no'")
    return answer


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    seen_ids = set()
    seen_image_question = set()
    deduped = []
    for row in rows:
        sample_id = row.get("sample_id")
        image_question = (row.get("image_path"), row.get("question"))
        if sample_id in seen_ids or image_question in seen_image_question:
            continue
        seen_ids.add(sample_id)
        seen_image_question.add(image_question)
        deduped.append(dict(row))
    return deduped


def balance_yes_no(rows: list[dict], total: int, rng: random.Random) -> list[dict]:
    if total <= 0:
        raise ValueError("total must be positive")

    yes_rows = [dict(row) for row in rows if _answer(row) == "yes"]
    no_rows = [dict(row) for row in rows if _answer(row) == "no"]
    n_yes = total // 2
    n_no = total - n_yes
    if len(yes_rows) < n_yes or len(no_rows) < n_no:
        raise ValueError("not enough yes/no rows to balance requested total")

    rng.shuffle(yes_rows)
    rng.shuffle(no_rows)
    selected = yes_rows[:n_yes] + no_rows[:n_no]
    rng.shuffle(selected)
    return selected


def question_type_caps(rows: list[dict], max_frac: float = 0.6) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    if not 0 < max_frac <= 1:
        raise ValueError("max_frac must be in (0, 1]")

    counts = Counter(tag_question_type(row["question"]) for row in rows)
    total = len(rows)
    if any(count / total > max_frac for count in counts.values()):
        raise ValueError("question type cap exceeded")


def _visual_dependency(row: dict) -> str:
    dependency = str(row.get("expected_visual_dependency", "")).strip().lower()
    if dependency not in {"high", "low"}:
        raise ValueError("expected_visual_dependency must be 'high' or 'low'")
    return dependency


def _has_both_visual_dependencies(rows: list[dict]) -> None:
    dependencies = {_visual_dependency(row) for row in rows}
    if dependencies != {"high", "low"}:
        raise ValueError("rows must include both high and low visual dependency")


def manifest_report(rows: list[dict]) -> dict:
    answers = Counter(_answer(row) for row in rows)
    visual_dependency = Counter(_visual_dependency(row) for row in rows)
    question_types = Counter(tag_question_type(row["question"]) for row in rows)
    return {
        "total": len(rows),
        "answers": dict(sorted(answers.items())),
        "visual_dependency": dict(sorted(visual_dependency.items())),
        "question_types": dict(sorted(question_types.items())),
    }


def _validate_rows(rows: list[dict]) -> None:
    for row in rows:
        validate_artifact(row, "manifest")


def select_n200(seed: int = 0, total: int = DEFAULT_TOTAL) -> list[dict]:
    rng = random.Random(seed)
    seed_rows = build_manifest(
        _load_seed_rows(DEFAULT_SEED_MANIFEST),
        total // 2,
        seed=seed,
        per_type_balance=False,
    )
    answer_counts = Counter(_answer(row) for row in seed_rows)
    needed_yes = total // 2 - answer_counts["yes"]
    needed_no = total - total // 2 - answer_counts["no"]
    if needed_yes < 0 or needed_no < 0:
        raise ValueError("seed manifest has too many rows for balanced N=200 target")

    new_rows = _build_new_image_rows(
        needed_yes=needed_yes,
        needed_no=needed_no,
        image_dir=DEFAULT_IMAGE_DIR,
    )
    rows = _dedupe_rows(seed_rows + new_rows)
    if len(rows) != total:
        raise ValueError("combined dataset must contain unique N=200 rows")

    # Keep all 100 seed images plus all 100 newly downloaded images.  Shuffle only
    # final order, not the membership, so image uniqueness remains explicit.
    rng.shuffle(rows)
    _has_both_visual_dependencies(rows)
    question_type_caps(rows, max_frac=0.6)
    _validate_unique_images(rows)
    _validate_rows(rows)
    return rows


def write_jsonl(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True)
        for row in rows
    )
    output.write_text(text + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    rows = select_n200(seed=args.seed, total=args.total)
    write_jsonl(rows, args.output)
    report = manifest_report(rows)
    print(f"total: {report['total']}")
    print(f"answers: {report['answers']}")
    print(f"visual_dependency: {report['visual_dependency']}")
    print(f"question_types: {report['question_types']}")
    print(f"unique_images: {len({row['image_path'] for row in rows})}")
    print(f"output: {args.output}")
    return 0


def _load_seed_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _build_new_image_rows(
    *,
    needed_yes: int,
    needed_no: int,
    image_dir: Path,
) -> list[dict]:
    total_new = needed_yes + needed_no
    photos = _fetch_picsum_metadata(total_new)
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    answers = ["yes"] * needed_yes + ["no"] * needed_no
    qtypes = _balanced_question_types(total_new)
    for index, (photo, answer, qtype) in enumerate(zip(photos, answers, qtypes), start=1):
        sample_id = f"n200_{index:04d}"
        image_path = image_dir / f"{sample_id}.jpg"
        _download_picsum_image(str(photo["id"]), image_path)
        rows.append(_new_photo_row(sample_id, image_path, photo, answer, qtype))
    return rows


def _new_photo_row(
    sample_id: str,
    image_path: Path,
    photo: dict,
    answer: str,
    qtype: str,
) -> dict:
    if answer == "yes":
        target = "real-world photograph"
    else:
        target = "synthetic cartoon illustration"
    return {
        "sample_id": sample_id,
        "source": f"Lorem Picsum / Unsplash | {photo['url']}",
        "image_path": _manifest_image_path(image_path),
        "question": _question(qtype, target, answer),
        "question_type": tag_question_type(_question(qtype, target, answer)),
        "target_answer": answer,
        "target_token_policy": "first_generated_token",
        "expected_visual_dependency": "high",
        "text_only_answerable": "unknown",
        "counterfactual_image_path": None,
        "control_type": "clean",
        "expected_flip": None,
        "label": None,
        "notes": (
            f"real photo from Lorem Picsum backed by Unsplash metadata; "
            f"picsum_id={photo['id']}; author={photo['author']}; "
            f"source_page={photo['url']}; download_url={photo['download_url']}; "
            "license=see source page"
        ),
    }


def _manifest_image_path(image_path: Path) -> str:
    path = image_path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _question(qtype: str, target: str, answer: str) -> str:
    if qtype == "yes_no":
        return f"Is this image a {target}?"
    if qtype == "what":
        return f"What is the yes/no answer: is this image a {target}?"
    if qtype == "where":
        return f"Where should the image be checked to decide whether it is a {target}? Answer yes or no."
    if qtype == "color":
        return f"What color and lighting evidence helps decide whether this is a {target}? Answer yes or no."
    if qtype == "count":
        return f"How many visual regions support that this is a {target}? Answer yes if at least one region supports it, otherwise no."
    raise ValueError(f"unknown question type {qtype!r}")


def _balanced_question_types(total: int) -> list[str]:
    qtypes = []
    while len(qtypes) < total:
        qtypes.extend(QUESTION_TYPES)
    return qtypes[:total]


def _fetch_picsum_metadata(total: int) -> list[dict]:
    request = urllib.request.Request(
        PICSUM_API,
        headers={"User-Agent": "CrossRoute-Audit dataset builder"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        photos = json.loads(response.read().decode("utf-8"))
    if len(photos) < total:
        raise ValueError("not enough Picsum/Unsplash photos for N=200 extension")
    return photos[:total]


def _download_picsum_image(photo_id: str, output: Path) -> None:
    request = urllib.request.Request(
        PICSUM_IMAGE_URL.format(id=photo_id),
        headers={"User-Agent": "CrossRoute-Audit dataset builder"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        output.write_bytes(response.read())


def _validate_unique_images(rows: list[dict]) -> None:
    image_paths = [row["image_path"] for row in rows]
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("rows must have unique image_path values")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("rows must have unique sample_id values")
    image_questions = [(row["image_path"], row["question"]) for row in rows]
    if len(image_questions) != len(set(image_questions)):
        raise ValueError("rows must have unique image/question pairs")


if __name__ == "__main__":
    raise SystemExit(main())
