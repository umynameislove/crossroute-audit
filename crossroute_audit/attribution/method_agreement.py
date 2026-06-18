"""Gradient x activation as the secondary attribution method."""
from __future__ import annotations

import math

from captum.attr import LayerGradientXActivation
import torch

from crossroute_audit.metrics.attribution_mass import attribution_mass_by_layer


def gradient_x_activation_mass(
    adapter,
    inputs,
    target_answer: str,
    policy: str,
) -> dict[str, dict[int, float]]:
    """Return secondary-method AttributionMass on the full audit layer axis.

    Gradient x activation uses the same fixed target token, audit layer
    outputs, token groups, and signed-token aggregation as Layer-IG. The method
    is intentionally lightweight enough to run on a small evidence subset.
    """
    prepared = adapter.prepare_attribution_inputs(inputs, target_answer, policy)
    if not isinstance(prepared, tuple) or len(prepared) != 3:
        raise TypeError(
            "prepare_attribution_inputs must return "
            "(encoder_embeddings, attention_mask, target_token_id)"
        )
    encoder_embeddings, attention_mask, target_token_id = prepared
    if not torch.is_tensor(encoder_embeddings) or encoder_embeddings.ndim != 3:
        raise ValueError(
            "encoder embeddings must have shape [batch, sequence, hidden]"
        )
    if encoder_embeddings.shape[0] != 1:
        raise ValueError("gradient x activation currently supports one sample at a time")
    if not torch.is_tensor(attention_mask) or attention_mask.shape != encoder_embeddings.shape[:2]:
        raise ValueError(
            "attention mask must match encoder embedding batch and sequence dimensions"
        )

    layer_count = int(adapter.get_intervention_layer_count())
    if layer_count <= 0:
        raise ValueError("intervention layer count must be positive")

    token_attribution_by_layer = {}
    for layer in range(layer_count):
        layer_embeddings = encoder_embeddings.detach().requires_grad_(True)
        with adapter.attribution_layer_output(layer) as attribution_layer:
            algorithm = LayerGradientXActivation(
                adapter.forward_target_logit_from_embeddings,
                attribution_layer,
                multiply_by_inputs=True,
            )
            attribution = algorithm.attribute(
                layer_embeddings,
                additional_forward_args=(
                    attention_mask,
                    int(target_token_id),
                ),
                attribute_to_layer_input=False,
            )
        token_attribution_by_layer[layer] = _token_attribution(
            _primary_attribution_tensor(attribution)
        )

    return attribution_mass_by_layer(
        token_attribution_by_layer,
        adapter.get_token_groups(inputs),
        layer_count,
    )


def _primary_attribution_tensor(attribution) -> torch.Tensor:
    if torch.is_tensor(attribution):
        return attribution
    if (
        isinstance(attribution, (tuple, list))
        and attribution
        and torch.is_tensor(attribution[0])
    ):
        return attribution[0]
    raise TypeError("Captum attribution must contain a tensor")


def _token_attribution(attribution: torch.Tensor) -> tuple[float, ...]:
    if attribution.ndim != 3 or attribution.shape[0] != 1:
        raise ValueError(
            "layer attribution must have shape [1, sequence, hidden], "
            f"got {list(attribution.shape)}"
        )
    scores = attribution.detach().to(dtype=torch.float32).sum(dim=-1)[0]
    values = tuple(float(value) for value in scores.cpu().tolist())
    if not all(math.isfinite(value) for value in values):
        raise ValueError("token attribution contains non-finite values")
    return values
