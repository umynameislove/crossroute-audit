from __future__ import annotations

import csv
import json
from pathlib import Path

from crossroute_audit.cli import build_parser
from crossroute_audit.synthetic.benchmark import FAULT_CLASSES


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_manifest(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )


def _sample(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "image_path": f"{sample_id}.jpg",
        "question": "Is the target visible?",
        "target_answer": "yes",
        "target_token_policy": "first_generated_token",
        "expected_visual_dependency": "high",
        "text_only_answerable": "no",
        "counterfactual_image_path": None,
        "control_type": "clean",
        "expected_flip": None,
        "label": None,
        "notes": "CLI batch fixture.",
    }


def test_validate_writes_benchmark(tmp_path):
    out = tmp_path / "val"
    args = build_parser().parse_args(["validate", "--out", str(out), "--n", "10"])

    assert args.n == 10, f"validate parser lost --n value: args={args}"
    assert args.func(args) == 0, "validate command should return shell success code 0"
    csv_path = out / "benchmark.csv"
    assert csv_path.is_file(), f"validate did not create benchmark CSV at {csv_path}"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    expected_rows = 10 * len(FAULT_CLASSES)
    assert len(rows) == expected_rows, (
        f"validate wrote the wrong number of benchmark rows: "
        f"expected={expected_rows}, got={len(rows)}, rows={rows[:3]}"
    )
    bad_rows = [row for row in rows if row["correct"] != "True"]
    assert not bad_rows, f"validate benchmark produced detector mistakes: {bad_rows[:5]}"


def test_report_writes_markdown(tmp_path):
    run = tmp_path / "audit"
    run.mkdir()
    _write_json(
        run / "audit_report_s1.json",
        {
            "sample_id": "s1",
            "diagnosis": {"diagnosis": "no_flag"},
            "rank_alignment": {"image": 0.5, "text": 0.4},
        },
    )
    out = tmp_path / "report.md"
    args = build_parser().parse_args(
        ["report", "--run", str(run), "--out", str(out)]
    )

    assert args.run == str(run), f"report parser lost --run value: args={args}"
    assert args.out == str(out), f"report parser lost --out value: args={args}"
    assert args.func(args) == 0, "report command should return shell success code 0"
    assert out.is_file(), f"report command did not create markdown file at {out}"
    text = out.read_text(encoding="utf-8")
    assert "s1" in text, f"report markdown omitted sample id; content={text!r}"
    expected_line = "| s1 | no_flag | 0.500 | 0.400 |"
    assert expected_line in text, (
        f"report markdown row does not match results_table formatting: "
        f"expected line={expected_line!r}, content={text!r}"
    )


def test_batch_writes_audit_reports(tmp_path):
    sample_id = "s1"
    manifest = tmp_path / "manifest.jsonl"
    control_dir = tmp_path / "control"
    causal_dir = tmp_path / "causal"
    attr_dir = tmp_path / "attr"
    out = tmp_path / "audit"

    _write_manifest(manifest, [_sample(sample_id)])
    _write_json(
        control_dir / f"control_status_{sample_id}.json",
        {
            "sample_id": sample_id,
            "target_answer": "yes",
            "which": ["text_only", "negative_control"],
            "controls": {
                "text_only": {"target_logit": 0.1, "answerable": "no"},
                "negative_control": {"effect": 0.0},
            },
        },
    )
    _write_json(
        causal_dir / f"causal_effect_{sample_id}.json",
        {
            "sample_id": sample_id,
            "target_answer": "yes",
            "C_by_layer": {
                "image": {"0": 0.1, "1": 0.2, "2": 0.3},
                "text": {"0": 0.1, "1": 0.2, "2": 0.3},
            },
            "stability": {},
            "settings": {},
        },
    )
    _write_json(
        attr_dir / f"attribution_mass_{sample_id}.json",
        {
            "sample_id": sample_id,
            "target_answer": "yes",
            "attribution_mass": {
                "image": {"0": 1.6, "1": 1.4, "2": 1.2},
                "text": {"0": 0.1, "1": 0.2, "2": 0.3},
            },
            "settings": {},
        },
    )

    args = build_parser().parse_args(
        [
            "batch",
            "--manifest",
            str(manifest),
            "--control-dir",
            str(control_dir),
            "--causal-dir",
            str(causal_dir),
            "--attr-dir",
            str(attr_dir),
            "--out",
            str(out),
        ]
    )

    assert args.manifest == str(manifest), f"batch parser lost --manifest: args={args}"
    assert args.control_dir == str(control_dir), (
        f"batch parser lost --control-dir: args={args}"
    )
    assert args.causal_dir == str(causal_dir), (
        f"batch parser lost --causal-dir: args={args}"
    )
    assert args.attr_dir == str(attr_dir), f"batch parser lost --attr-dir: args={args}"
    assert args.out == str(out), f"batch parser lost --out: args={args}"
    assert args.func(args) == 0, "batch command should return shell success code 0"
    report_path = out / f"audit_report_{sample_id}.json"
    assert report_path.is_file(), f"batch did not create audit report at {report_path}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["sample_id"] == sample_id, (
        f"batch report sample_id mismatch: expected={sample_id}, report={report}"
    )
    assert report["rank_alignment"]["image"] == -1.0, (
        f"batch should preserve string layer keys and compute inverse image rank; "
        f"rank_alignment={report['rank_alignment']}, report={report}"
    )
    expected_diagnosis = "false_attribution_persistence"
    got_diagnosis = report["diagnosis"]["diagnosis"]
    assert got_diagnosis == expected_diagnosis, (
        f"batch diagnosis mismatch: expected={expected_diagnosis!r}, "
        f"got={got_diagnosis!r}, report={report}"
    )
