"""CrossRoute-Audit command-line interface.

Commands:
    crossroute audit    --model blip2 --image sample.jpg --question "Is there a dog?" \
                        --target yes --controls text_only,no_image,counterfactual --out result.json
    crossroute batch    --manifest samples.jsonl --control-dir runs/control \
                        --causal-dir runs/causal --attr-dir runs/attr --out runs/audit/
    crossroute validate --out validation/ --n 40
    crossroute report   --run runs/audit/ --out report.md
"""
from __future__ import annotations

import argparse


def cmd_audit(args):
    raise NotImplementedError("audit: produce a per-sample audit report")


def cmd_batch(args):
    from crossroute_audit.io.audit_report import audit_report_for_manifest

    paths = audit_report_for_manifest(
        args.manifest,
        args.control_dir,
        args.causal_dir,
        args.attr_dir,
        args.out,
    )
    print(f"wrote {len(paths)} audit reports -> {args.out}")
    return 0


def cmd_validate(args):
    from pathlib import Path

    from crossroute_audit.synthetic.benchmark import run_benchmark

    summary = run_benchmark(Path(args.out) / "benchmark.csv", n_per_fault=args.n)
    print(
        f"validate: accuracy={summary['accuracy']:.3f} over {summary['total']} cases "
        f"-> {args.out}/benchmark.csv"
    )
    return 0


def cmd_report(args):
    import glob
    import json
    from pathlib import Path

    from crossroute_audit.io.report import results_table

    files = sorted(glob.glob(str(Path(args.run) / "audit_report_*.json")))
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in files]
    markdown = results_table(reports)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown + "\n", encoding="utf-8")
    print(f"report: wrote {args.out} ({len(reports)} samples)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crossroute",
        description="CrossRoute-Audit: explanation-faithfulness auditing for vision-language models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Audit a single sample")
    audit.add_argument("--model", default="blip2")
    audit.add_argument("--image", required=True)
    audit.add_argument("--question", required=True)
    audit.add_argument("--target", required=True)
    audit.add_argument("--controls", default="text_only,no_image,counterfactual")
    audit.add_argument("--out", required=True)
    audit.set_defaults(func=cmd_audit)

    batch = sub.add_parser("batch", help="Audit a batch from a manifest")
    batch.add_argument("--manifest", required=True)
    batch.add_argument("--control-dir", required=True, dest="control_dir")
    batch.add_argument("--causal-dir", required=True, dest="causal_dir")
    batch.add_argument("--attr-dir", required=True, dest="attr_dir")
    batch.add_argument("--out", required=True)
    batch.set_defaults(func=cmd_batch)

    validate = sub.add_parser("validate", help="Run the synthetic fault suite")
    validate.add_argument("--suite", required=False)
    validate.add_argument("--n", type=int, default=40)
    validate.add_argument("--out", required=True)
    validate.set_defaults(func=cmd_validate)

    report = sub.add_parser("report", help="Summarize a run into a report")
    report.add_argument("--run", required=True)
    report.add_argument("--out", required=True)
    report.set_defaults(func=cmd_report)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
