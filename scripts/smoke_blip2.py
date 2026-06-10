"""Run a manual BLIP-2 forward-path smoke test on a CUDA machine."""
from __future__ import annotations

import argparse
import json

from crossroute_audit.model_adapters.blip2_adapter import BLIP2Adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--policy",
        default="selected_candidate_logit",
        choices=[
            "exact_token",
            "first_generated_token",
            "selected_candidate_logit",
        ],
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    adapter = BLIP2Adapter(device=args.device)
    inputs = adapter.prepare_inputs(args.image, args.question)
    output = adapter.forward(inputs, capture=True)
    target_logit = adapter.get_target_logit(inputs, args.target, args.policy)
    token_groups = adapter.get_token_groups(inputs)

    print(
        json.dumps(
            {
                "forward": output.meta,
                "target_logit": target_logit,
                "qformer_layer_count": adapter.get_layer_count(),
                "token_group_sizes": {
                    "image": len(token_groups.image),
                    "text": len(token_groups.text),
                    "fusion": len(token_groups.fusion),
                    "answer": len(token_groups.answer),
                },
                "hidden_state_groups": {
                    key: len(value) for key, value in output.hidden_states.items()
                },
                "attention_groups": {
                    key: len(value) for key, value in output.attentions.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
