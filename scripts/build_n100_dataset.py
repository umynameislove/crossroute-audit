"""Build the CrossRoute-Audit VQA yes/no manifest.

By default the builder writes only the 70 new VQA yes/no samples under
``data/images/vqa_visual/``.  The older 30 pilot images are intentionally not
included unless ``--include-pilot`` is supplied for legacy N=100 regeneration.
Unit tests exercise the pure helpers only; running the CLI may download dataset
rows and images.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossroute_audit.io.dataset_loader import build_manifest, tag_question_type
from crossroute_audit.io.schema import validate_artifact


DEFAULT_PILOT_MANIFEST = REPO_ROOT / "data" / "manifest" / "samples.jsonl"
DEFAULT_OUTPUT_MANIFEST = REPO_ROOT / "data" / "manifest" / "samples_n100.jsonl"
DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "images" / "vqa_visual"
DEFAULT_DATASET_NAME = "Multimodal-Fatima/VQAv2_sample_train"
DEFAULT_SPLIT = "train"
DEFAULT_TOTAL = 70
DEFAULT_WITH_PILOT_TOTAL = 100
DEFAULT_NEW_LIMIT = 70


def normalize_target_answer(answer: Any) -> str | None:
    """Return normalized yes/no answer, or None for any non yes/no answer."""
    if answer is None:
        return None
    normalized = str(answer).strip().lower().strip(" \t\r\n.,!?;:'\"")
    if normalized in {"yes", "no"}:
        return normalized
    return None


def repo_relative_path(path: Path | str, repo_root: Path = REPO_ROOT) -> str:
    """Return a POSIX path relative to the repository root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside repository root: {candidate}") from exc


def resolve_repo_path(path: Path | str, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve absolute or repository-relative paths."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def read_jsonl(path: Path | str) -> list[dict]:
    """Read a JSONL file as a list of dictionaries."""
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(record)
    return records


def make_vqa_record(
    *,
    sequence_id: int,
    image_path: Path | str,
    question: str,
    target_answer: Any,
    dataset_name: str,
    split: str,
    source_index: int | str,
    source_metadata: dict[str, Any] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict:
    """Build one schema-valid manifest record for a yes/no VQA sample."""
    answer = normalize_target_answer(target_answer)
    if answer is None:
        raise ValueError(f"target_answer must normalize to yes/no: {target_answer!r}")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    relative_image_path = repo_relative_path(Path(image_path), repo_root=repo_root)
    if Path(relative_image_path).is_absolute():
        raise ValueError("image_path must be relative to the repository root")

    metadata = source_metadata or {}
    note_parts = [
        f"dataset={dataset_name}",
        f"split={split}",
        f"source_index={source_index}",
    ]
    for key in ("question_id", "image_id", "url", "image_url", "license", "creator"):
        value = metadata.get(key)
        if value not in (None, ""):
            note_parts.append(f"{key}={value}")

    record = {
        "sample_id": f"vqa_{sequence_id:04d}",
        "source": f"HuggingFace datasets | {dataset_name} | {split}",
        "image_path": relative_image_path,
        "question": " ".join(question.strip().split()),
        "question_type": tag_question_type(question),
        "target_answer": answer,
        "target_token_policy": "exact_token",
        "expected_visual_dependency": "high",
        "text_only_answerable": "unknown",
        "control_type": "clean",
        "notes": "; ".join(note_parts),
    }
    validate_artifact(record, "manifest")
    return record


def build_n100_records(
    pilot_records: list[dict],
    new_records: list[dict],
    *,
    total: int = DEFAULT_TOTAL,
    seed: int = 0,
) -> list[dict]:
    """Merge pilot and new records, then validate through build_manifest."""
    if len(pilot_records) > total:
        raise ValueError(
            f"pilot record count {len(pilot_records)} exceeds requested total {total}"
        )

    needed_new = total - len(pilot_records)
    if len(new_records) < needed_new:
        raise ValueError(
            f"need {needed_new} new records, but only {len(new_records)} were provided"
        )

    selected_new = deterministic_sample(new_records, needed_new, seed=seed)
    combined = [dict(record) for record in pilot_records] + [
        dict(record) for record in selected_new
    ]
    _assert_unique_field(combined, "sample_id")
    _assert_unique_field(combined, "image_path")

    manifest = build_manifest(combined, total, seed=seed, per_type_balance=False)
    _assert_unique_field(manifest, "sample_id")
    _assert_unique_field(manifest, "image_path")
    return manifest


def build_new_only_records(new_records: list[dict], *, total: int = DEFAULT_TOTAL) -> list[dict]:
    """Return sorted VQA-only records, one per vqa_visual image."""
    if len(new_records) < total:
        raise ValueError(f"need {total} new records, but only {len(new_records)} provided")

    records_by_path: dict[str, dict] = {}
    for record in new_records:
        image_path = record.get("image_path")
        if not isinstance(image_path, str):
            raise ValueError("new record image_path must be a string")
        if not image_path.startswith("data/images/vqa_visual/"):
            raise ValueError(f"new record image_path is outside vqa_visual: {image_path}")
        records_by_path[image_path] = dict(record)

    selected: list[dict] = []
    for sequence_id in range(1, total + 1):
        sample_id = f"vqa_{sequence_id:04d}"
        image_path = f"data/images/vqa_visual/{sample_id}.jpg"
        if image_path not in records_by_path:
            raise ValueError(f"missing metadata for {image_path}")
        record = records_by_path[image_path]
        if record.get("sample_id") != sample_id:
            raise ValueError(
                f"sample_id mismatch for {image_path}: {record.get('sample_id')}"
            )
        record["question_type"] = record.get("question_type") or tag_question_type(
            record["question"]
        )
        validate_artifact(record, "manifest")
        selected.append(record)

    _assert_unique_field(selected, "sample_id")
    _assert_unique_field(selected, "image_path")
    return selected


def deterministic_sample(records: list[dict], n: int, *, seed: int) -> list[dict]:
    """Return a deterministic sample without mutating the input list."""
    if n > len(records):
        raise ValueError(f"requested {n} records from only {len(records)} records")
    rng = random.Random(seed)
    return rng.sample([dict(record) for record in records], n)


def write_manifest_atomic(
    records: list[dict],
    output_manifest: Path | str,
    *,
    pilot_manifest: Path | str = DEFAULT_PILOT_MANIFEST,
) -> None:
    """Write JSONL atomically, refusing to overwrite the pilot manifest."""
    output_path = Path(output_manifest)
    pilot_path = Path(pilot_manifest)
    if output_path.resolve() == pilot_path.resolve():
        raise ValueError("refusing to overwrite data/manifest/samples.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_artifact(record, "manifest")
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    tmp_path.replace(output_path)


def collect_hf_yes_no_records(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    limit: int,
    seed: int,
    image_dir: Path,
    repo_root: Path = REPO_ROOT,
    shuffle_buffer: int = 10_000,
    max_scan: int = 50_000,
    streaming: bool = True,
) -> list[dict]:
    """Collect schema-valid yes/no VQA records from a HuggingFace dataset."""
    dataset = _load_hf_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        seed=seed,
        shuffle_buffer=shuffle_buffer,
        streaming=streaming,
    )
    image_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    image_errors: list[str] = []
    for source_index, sample in enumerate(dataset):
        if source_index >= max_scan:
            break
        if len(records) == limit:
            break
        if not isinstance(sample, dict):
            continue

        question = extract_question(sample)
        answer = extract_yes_no_answer(sample)
        if question is None or answer is None:
            continue
        if tag_question_type(question) != "yes_no":
            continue

        sequence_id = len(records) + 1
        image_path = image_dir / f"vqa_{sequence_id:04d}.jpg"
        try:
            save_sample_image(sample, image_path)
        except Exception as exc:  # pragma: no cover - depends on external dataset
            if len(image_errors) < 5:
                image_errors.append(f"index={source_index}: {exc}")
            continue

        record = make_vqa_record(
            sequence_id=sequence_id,
            image_path=image_path,
            question=question,
            target_answer=answer,
            dataset_name=dataset_name,
            split=split,
            source_index=source_index,
            source_metadata=source_metadata(sample),
            repo_root=repo_root,
        )
        records.append(record)

    if len(records) < limit:
        detail = "; ".join(image_errors) if image_errors else "no image errors captured"
        raise RuntimeError(
            f"collected {len(records)} valid yes/no samples, need {limit}; "
            f"scanned up to {max_scan} rows; image errors: {detail}"
        )
    return records


def extract_question(sample: dict[str, Any]) -> str | None:
    """Extract a non-empty question string from common VQA dataset fields."""
    for key in ("question", "question_text", "prompt"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def extract_yes_no_answer(sample: dict[str, Any]) -> str | None:
    """Extract a yes/no answer from common VQA dataset answer formats."""
    for key in ("target_answer", "answer", "multiple_choice_answer", "label"):
        answer = normalize_target_answer(sample.get(key))
        if answer is not None:
            return answer

    answers = sample.get("answers")
    if not isinstance(answers, list):
        return None

    counts: collections.Counter[str] = collections.Counter()
    for item in answers:
        if isinstance(item, dict):
            raw_answer = item.get("answer")
        else:
            raw_answer = item
        normalized = normalize_target_answer(raw_answer)
        if normalized is not None:
            counts[normalized] += 1

    if not counts:
        return None
    top_count = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == top_count]
    if len(winners) != 1:
        return None
    return winners[0]


def source_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    """Keep only factual source metadata fields supported by notes/source."""
    metadata: dict[str, Any] = {}
    for key in (
        "question_id",
        "image_id",
        "id_image",
        "url",
        "image_url",
        "source_url",
        "license",
        "creator",
        "author",
    ):
        value = sample.get(key)
        if isinstance(value, (str, int, float)) and value != "":
            if key == "source_url":
                metadata["url"] = value
            elif key == "id_image":
                metadata["image_id"] = value
            else:
                metadata[key] = value
        elif isinstance(value, list) and value:
            metadata[key] = value[0]
    if "author" in metadata and "creator" not in metadata:
        metadata["creator"] = metadata.pop("author")
    return metadata


def save_sample_image(sample: dict[str, Any], out_path: Path) -> None:
    """Save an image from a HuggingFace sample as RGB JPEG."""
    image_value = None
    for key in ("image", "img", "image_file"):
        if key in sample:
            image_value = sample[key]
            break
    if image_value is None:
        raise ValueError("sample has no image field")

    image = _coerce_to_pil_image(image_value)
    image = ImageOps.exif_transpose(image).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="JPEG", quality=95)


def _coerce_to_pil_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("bytes"), bytes):
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return _coerce_to_pil_image(value["path"])
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value))
    if isinstance(value, (str, Path)):
        text = str(value)
        if text.startswith(("http://", "https://")):
            with urllib.request.urlopen(text, timeout=30) as response:
                return Image.open(io.BytesIO(response.read()))
        return Image.open(text)
    raise ValueError(f"unsupported image value type: {type(value).__name__}")


def validate_image_paths(records: Iterable[dict], *, repo_root: Path = REPO_ROOT) -> None:
    """Raise if any manifest image path is absolute or missing on disk."""
    missing: list[str] = []
    absolute: list[str] = []
    for record in records:
        image_path = record["image_path"]
        if Path(image_path).is_absolute():
            absolute.append(image_path)
            continue
        if not (repo_root / image_path).is_file():
            missing.append(image_path)
    if absolute:
        raise ValueError(f"manifest contains absolute image_path values: {absolute[:5]}")
    if missing:
        raise FileNotFoundError(f"manifest image files missing: {missing[:5]}")


def _load_hf_dataset(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    seed: int,
    shuffle_buffer: int,
    streaming: bool,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "HuggingFace datasets is required to build the VQA manifest. "
            "Install it with `python -m pip install datasets`."
        ) from exc

    args = [dataset_name]
    if dataset_config:
        args.append(dataset_config)
    dataset = load_dataset(*args, split=split, streaming=streaming)
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return dataset


def _assert_unique_field(records: list[dict], field: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        value = record[field]
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"duplicate {field} values: {duplicates[:5]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-new", type=int, default=DEFAULT_NEW_LIMIT)
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--pilot-manifest", default=str(DEFAULT_PILOT_MANIFEST))
    parser.add_argument("--output-manifest", default=str(DEFAULT_OUTPUT_MANIFEST))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--max-scan", type=int, default=50_000)
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Load a normal split instead of a streaming split.",
    )
    parser.add_argument(
        "--strict-image-paths",
        action="store_true",
        help="Fail if any image_path in the final manifest is missing.",
    )
    parser.add_argument(
        "--include-pilot",
        action="store_true",
        help=(
            "Legacy mode: include data/manifest/samples.jsonl pilot rows and "
            "write a 100-row mixed manifest."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT
    pilot_manifest = resolve_repo_path(args.pilot_manifest, repo_root=repo_root)
    output_manifest = resolve_repo_path(args.output_manifest, repo_root=repo_root)
    image_dir = resolve_repo_path(args.image_dir, repo_root=repo_root)

    pilot_records: list[dict] = []
    collect_limit = args.limit_new
    if args.include_pilot:
        pilot_records = read_jsonl(pilot_manifest)
        total = args.total if args.total != DEFAULT_TOTAL else DEFAULT_WITH_PILOT_TOTAL
        needed_new = total - len(pilot_records)
        if needed_new < 0:
            raise ValueError(
                f"pilot manifest has {len(pilot_records)} rows, more than total={total}"
            )
        if args.limit_new < needed_new:
            raise ValueError(
                f"--limit-new must be at least {needed_new}; got {args.limit_new}"
            )
        collect_limit = args.limit_new
    else:
        total = args.total
        if total != args.limit_new:
            raise ValueError(
                "VQA-only mode expects --total to match --limit-new so manifest "
                "rows map exactly to vqa_0001..vqa_N"
            )

    new_records = collect_hf_yes_no_records(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        limit=collect_limit,
        seed=args.seed,
        image_dir=image_dir,
        repo_root=repo_root,
        shuffle_buffer=args.shuffle_buffer,
        max_scan=args.max_scan,
        streaming=not args.no_streaming,
    )
    if args.include_pilot:
        manifest = build_n100_records(
            pilot_records,
            new_records,
            total=total,
            seed=args.seed,
        )
    else:
        manifest = build_new_only_records(new_records, total=total)

    if args.strict_image_paths:
        validate_image_paths(manifest, repo_root=repo_root)

    write_manifest_atomic(manifest, output_manifest, pilot_manifest=pilot_manifest)
    question_type_counts = collections.Counter(
        record.get("question_type", "unknown") for record in manifest
    )
    target_counts = collections.Counter(record["target_answer"] for record in manifest)
    print(f"wrote_manifest={repo_relative_path(output_manifest, repo_root=repo_root)}")
    print(f"total={len(manifest)}")
    print(f"pilot={len(pilot_records)}")
    print(f"new={len(manifest) - len(pilot_records)}")
    print(f"question_type_distribution={dict(sorted(question_type_counts.items()))}")
    print(f"target_answer_distribution={dict(sorted(target_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
