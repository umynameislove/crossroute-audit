"""CrossRoute-Audit command-line interface.

Commands:
    crossroute audit    --model blip2 --image sample.jpg --question "Is there a dog?" \
                        --target yes --controls text_only,no_image,counterfactual --out result.json
    crossroute batch    --manifest samples.jsonl --out runs/mvp_blip2/
    crossroute validate --suite synthetic_faults.yaml --out validation/
    crossroute report   --run validation/ --out report.md
"""
from __future__ import annotations

import argparse


def cmd_audit(args):
    raise NotImplementedError("audit: produce a per-sample audit report")


def cmd_batch(args):
    raise NotImplementedError("batch: run the audit across a manifest")


def cmd_validate(args):
    raise NotImplementedError("validate: run the synthetic fault suite")


def cmd_report(args):
    raise NotImplementedError("report: summarize a run into a short report")


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
    batch.add_argument("--out", required=True)
    batch.set_defaults(func=cmd_batch)

    validate = sub.add_parser("validate", help="Run the synthetic fault suite")
    validate.add_argument("--suite", required=True)
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
