"""Build deterministic, schema-valid manifests from raw dataset records."""
from __future__ import annotations

import random

from crossroute_audit.io.schema import validate_artifact


_QUESTION_TYPES = ("yes_no", "count", "color", "what", "where", "other")
_YES_NO_PREFIXES = (
    "is",
    "are",
    "does",
    "do",
    "can",
    "did",
    "was",
    "were",
    "has",
    "have",
    "will",
    "would",
    "could",
    "should",
)
_REQUIRED_RAW_FIELDS = ("image_path", "question", "target_answer")


def tag_question_type(question: str) -> str:
    """Classify a question using deterministic, case-insensitive prefix rules."""
    if not isinstance(question, str):
        raise ValueError("question must be a string")

    normalized = " ".join(question.strip().lower().split())
    if normalized.startswith("how many"):
        return "count"
    if normalized.startswith("what color"):
        return "color"
    if normalized.startswith("where"):
        return "where"
    if normalized.startswith(("what", "which")):
        return "what"
    if any(
        normalized == prefix or normalized.startswith(f"{prefix} ")
        for prefix in _YES_NO_PREFIXES
    ):
        return "yes_no"
    return "other"


def build_manifest(
    records: list[dict],
    n: int,
    *,
    seed: int = 0,
    per_type_balance: bool = True,
) -> list[dict]:
    """Return ``n`` deduplicated, deterministic, schema-valid manifest records."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a non-negative integer")

    prepared = _prepare_unique_records(records)
    if n > len(prepared):
        raise ValueError(
            f"requested n={n}, but only {len(prepared)} unique records are available"
        )

    rng = random.Random(seed)
    if per_type_balance:
        selected = _balanced_sample(prepared, n, rng)
    else:
        selected = rng.sample(prepared, n)

    rng.shuffle(selected)
    for record in selected:
        validate_artifact(record, "manifest")
    return selected


def _prepare_unique_records(records: list[dict]) -> list[dict]:
    if not isinstance(records, list):
        raise ValueError("records must be a list of dictionaries")

    unique: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"record at index {index} must be a dictionary")

        missing = [field for field in _REQUIRED_RAW_FIELDS if field not in raw_record]
        if missing:
            raise ValueError(
                f"record at index {index} is missing required fields: {missing}"
            )

        image_path = raw_record["image_path"]
        question = raw_record["question"]
        if not isinstance(image_path, str) or not isinstance(question, str):
            raise ValueError(
                f"record at index {index} must have string image_path and question"
            )

        dedup_key = (image_path, question)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        record = dict(raw_record)
        record.setdefault("sample_id", f"sample_{index:06d}")
        record.setdefault("target_token_policy", "first_generated_token")
        record.setdefault("expected_visual_dependency", "unknown")
        record.setdefault("text_only_answerable", "unknown")
        record.setdefault("control_type", "clean")
        record["question_type"] = tag_question_type(question)
        unique.append(record)

    return unique


def _balanced_sample(
    records: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    buckets = {question_type: [] for question_type in _QUESTION_TYPES}
    for record in records:
        buckets[record["question_type"]].append(record)

    active_types = [
        question_type
        for question_type in _QUESTION_TYPES
        if buckets[question_type]
    ]
    rng.shuffle(active_types)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[dict] = []
    while len(selected) < n:
        added_this_round = False
        for question_type in active_types:
            if len(selected) == n:
                break
            bucket = buckets[question_type]
            if bucket:
                selected.append(bucket.pop())
                added_this_round = True
        if not added_this_round:
            break

    return selected
