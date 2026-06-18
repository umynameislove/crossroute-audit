"""Layer Integrated Gradients on adapter-defined audit-layer hidden states."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from captum.attr import LayerIntegratedGradients
import torch
import yaml

from crossroute_audit.attribution.completeness import completeness_residual
from crossroute_audit.io.manifest import load_manifest
from crossroute_audit.metrics.attribution_mass import attribution_mass_by_layer


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
_SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_BASELINE_NAME = "blank_image_audit_embeddings"


@dataclass(frozen=True)
class LayerIGAttribution:
    """Serializable attribution summary for one audit layer."""

    layer: int
    token_attribution: tuple[float, ...]
    completeness_residual: float
    convergence_delta: float


@dataclass(frozen=True)
class LayerIGRun:
    """Serializable Layer-IG output across the full intervention layer axis."""

    layers: dict[int, LayerIGAttribution]
    target_token_id: int
    target_logit: float
    baseline_logit: float
    ig_steps: int


@dataclass(frozen=True)
class _AttributionContext:
    """Internal tensor context. It is never returned or serialized."""

    audit_embeddings: torch.Tensor
    baseline_embeddings: torch.Tensor
    attention_mask: torch.Tensor
    target_token_id: int
    target_logit: float
    baseline_logit: float


def load_ig_steps(config_path: str | Path | None = None) -> int:
    """Read and validate ``attribution.ig_steps`` from the MVP config."""
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"attribution config does not exist: {path}")
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    try:
        value = config["attribution"]["ig_steps"]
    except (KeyError, TypeError) as exc:
        raise ValueError("config must define attribution.ig_steps") from exc
    return _validate_ig_steps(value)


def layer_integrated_gradients(
    adapter,
    inputs,
    target_answer: str,
    policy: str,
    layer: int,
    *,
    n_steps: int | None = None,
    internal_batch_size: int = 1,
) -> LayerIGAttribution:
    """Return signed per-token Layer-IG attribution for one audit layer.

    Hidden-dimension attributions are summed into one signed score per token.
    AttributionMass later applies ``sum(abs(token_score))`` over image and text
    positions. The baseline is provided by the adapter.
    """
    steps = load_ig_steps() if n_steps is None else _validate_ig_steps(n_steps)
    batch_size = _validate_internal_batch_size(internal_batch_size)
    prepared = _prepare_attribution_inputs(
        adapter,
        inputs,
        target_answer,
        policy,
    )
    baseline_embeddings = adapter.attribution_baseline_embeddings(inputs)
    with adapter.attribution_float32():
        context = _prepare_context(adapter, prepared, baseline_embeddings)
        return _attribute_layer(
            adapter,
            context,
            layer,
            n_steps=steps,
            internal_batch_size=batch_size,
        )


def layer_integrated_gradients_all_layers(
    adapter,
    inputs,
    target_answer: str,
    policy: str,
    *,
    n_steps: int | None = None,
    internal_batch_size: int = 1,
) -> LayerIGRun:
    """Compute Layer-IG for every zero-based audit layer."""
    steps = load_ig_steps() if n_steps is None else _validate_ig_steps(n_steps)
    batch_size = _validate_internal_batch_size(internal_batch_size)
    layer_count = int(adapter.get_intervention_layer_count())
    if layer_count <= 0:
        raise ValueError("intervention layer count must be positive")

    prepared = _prepare_attribution_inputs(
        adapter,
        inputs,
        target_answer,
        policy,
    )
    baseline_embeddings = adapter.attribution_baseline_embeddings(inputs)
    with adapter.attribution_float32():
        context = _prepare_context(adapter, prepared, baseline_embeddings)
        layers = {}
        for layer in range(layer_count):
            layers[layer] = _attribute_layer(
                adapter,
                context,
                layer,
                n_steps=steps,
                internal_batch_size=batch_size,
            )
        run = LayerIGRun(
            layers=layers,
            target_token_id=context.target_token_id,
            target_logit=context.target_logit,
            baseline_logit=context.baseline_logit,
            ig_steps=steps,
        )
    return run


def build_attribution_mass_result(
    adapter,
    inputs,
    sample: dict,
    *,
    n_steps: int | None = None,
    internal_batch_size: int = 1,
) -> dict[str, Any]:
    """Build a tensor-free per-sample AttributionMass artifact."""
    sample_id = _validated_sample_id(sample["sample_id"])
    run = layer_integrated_gradients_all_layers(
        adapter,
        inputs,
        sample["target_answer"],
        sample["target_token_policy"],
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )
    layer_count = int(adapter.get_intervention_layer_count())
    token_groups = adapter.get_token_groups(inputs)
    token_attribution_by_layer = {
        layer: result.token_attribution for layer, result in run.layers.items()
    }
    mass = attribution_mass_by_layer(
        token_attribution_by_layer,
        token_groups,
        layer_count,
    )

    return {
        "sample_id": sample_id,
        "target_answer": sample["target_answer"],
        "target_token_id": run.target_token_id,
        "target_logit": run.target_logit,
        "baseline_logit": run.baseline_logit,
        "attribution_mass": {
            group: _layer_map_for_json(values) for group, values in mass.items()
        },
        "completeness_residual": _layer_map_for_json(
            {
                layer: result.completeness_residual
                for layer, result in run.layers.items()
            }
        ),
        "convergence_delta": _layer_map_for_json(
            {
                layer: result.convergence_delta
                for layer, result in run.layers.items()
            }
        ),
        "settings": {
            "method": "layer_integrated_gradients",
            "baseline": _BASELINE_NAME,
            "compute_dtype": "float32",
            "aggregation": "sum_abs_signed_token_attribution",
            "groups": ["image", "text"],
            "ig_steps": run.ig_steps,
            "internal_batch_size": int(internal_batch_size),
            "layer_axis": adapter.layer_axis_name(),
            "layer_count": layer_count,
        },
    }


def write_attribution_mass(result: dict, out_path: str | Path) -> None:
    """Atomically write deterministic JSON and reject non-finite values."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as out_file:
            temporary_path = Path(out_file.name)
            json.dump(
                result,
                out_file,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            out_file.write("\n")
            out_file.flush()
            os.fsync(out_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def attribution_mass_for_manifest(
    adapter,
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    n_steps: int | None = None,
    internal_batch_size: int = 1,
) -> list[str]:
    """Write one ``attribution_mass_<sample_id>.json`` per manifest sample."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sample in load_manifest(manifest_path):
        sample_id = _validated_sample_id(sample["sample_id"])
        inputs = adapter.prepare_inputs(sample["image_path"], sample["question"])
        result = build_attribution_mass_result(
            adapter,
            inputs,
            sample,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
        )
        out_path = output_dir / f"attribution_mass_{sample_id}.json"
        write_attribution_mass(result, out_path)
        paths.append(str(out_path))
    return paths


def _prepare_attribution_inputs(
    adapter,
    inputs,
    target_answer: str,
    policy: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prepared = adapter.prepare_attribution_inputs(inputs, target_answer, policy)
    if not isinstance(prepared, tuple) or len(prepared) != 3:
        raise TypeError(
            "prepare_attribution_inputs must return "
            "(embeddings, attention_mask, target_token_id)"
        )
    return prepared


def _prepare_context(
    adapter,
    prepared: tuple[torch.Tensor, torch.Tensor, int],
    baseline_embeddings_raw: torch.Tensor,
) -> _AttributionContext:
    audit_embeddings, attention_mask, target_token_id = prepared
    if not torch.is_tensor(audit_embeddings) or audit_embeddings.ndim != 3:
        raise ValueError(
            "audit embeddings must have shape [batch, sequence, hidden]"
        )
    if audit_embeddings.shape[0] != 1:
        raise ValueError("Layer-IG currently supports one sample at a time")
    if not torch.is_tensor(attention_mask) or attention_mask.shape != audit_embeddings.shape[:2]:
        raise ValueError(
            "attention mask must match audit embedding batch and sequence dimensions"
        )
    target_token_id = int(target_token_id)

    clean_embeddings = (
        audit_embeddings.detach()
        .to(dtype=torch.float32)
        .requires_grad_(True)
    )
    baseline_embeddings = _validated_baseline(
        baseline_embeddings_raw,
        clean_embeddings,
    )
    target_logit = _scalar_forward(
        adapter,
        clean_embeddings,
        attention_mask,
        target_token_id,
    )
    baseline_logit = _scalar_forward(
        adapter,
        baseline_embeddings,
        attention_mask,
        target_token_id,
    )
    return _AttributionContext(
        audit_embeddings=clean_embeddings,
        baseline_embeddings=baseline_embeddings,
        attention_mask=attention_mask.detach(),
        target_token_id=target_token_id,
        target_logit=target_logit,
        baseline_logit=baseline_logit,
    )


def _validated_baseline(
    baseline: torch.Tensor,
    clean_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Validate and cast the on-manifold IG baseline to the clean-input shape."""
    if not torch.is_tensor(baseline) or baseline.ndim != 3:
        raise ValueError(
            "attribution baseline must have shape [batch, sequence, hidden]"
        )
    if baseline.shape != clean_embeddings.shape:
        raise ValueError(
            "attribution baseline must match clean embedding shape "
            f"{list(clean_embeddings.shape)}, got {list(baseline.shape)}"
        )
    return baseline.detach().to(dtype=torch.float32)


def _attribute_layer(
    adapter,
    context: _AttributionContext,
    layer: int,
    *,
    n_steps: int,
    internal_batch_size: int,
) -> LayerIGAttribution:
    if not isinstance(layer, int):
        raise TypeError("layer must be an integer")
    layer_count = int(adapter.get_intervention_layer_count())
    if layer < 0 or layer >= layer_count:
        raise IndexError(
            f"audit layer {layer} is outside valid range [0, {layer_count - 1}]"
        )

    audit_embeddings = context.audit_embeddings.detach().requires_grad_(True)
    baseline_embeddings = context.baseline_embeddings.detach()
    with adapter.attribution_layer_output(layer) as attribution_layer:
        algorithm = LayerIntegratedGradients(
            adapter.forward_target_logit_from_embeddings,
            attribution_layer,
            multiply_by_inputs=True,
        )
        attribution, convergence_delta = algorithm.attribute(
            audit_embeddings,
            baselines=baseline_embeddings,
            additional_forward_args=(
                context.attention_mask,
                context.target_token_id,
            ),
            n_steps=n_steps,
            method="gausslegendre",
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True,
            attribute_to_layer_input=False,
        )

    attribution_tensor = _primary_attribution_tensor(attribution)
    token_attribution = _token_attribution(attribution_tensor)
    convergence_value = _single_finite_value(
        convergence_delta,
        "convergence_delta",
    )
    residual = completeness_residual(
        sum(token_attribution),
        context.target_logit,
        context.baseline_logit,
    )
    return LayerIGAttribution(
        layer=layer,
        token_attribution=token_attribution,
        completeness_residual=residual,
        convergence_delta=convergence_value,
    )


def _scalar_forward(
    adapter,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_id: int,
) -> float:
    output = adapter.forward_target_logit_from_embeddings(
        embeddings,
        attention_mask,
        target_token_id,
    )
    return _single_finite_value(output, "target_logit")


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
    token_scores = attribution.detach().to(dtype=torch.float32).sum(dim=-1)[0]
    if not bool(torch.isfinite(token_scores).all()):
        raise ValueError("token attribution contains non-finite values")
    return tuple(float(value) for value in token_scores.cpu().tolist())


def _single_finite_value(value, name: str) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain exactly one value")
        scalar = float(value.detach().to(dtype=torch.float32).cpu().item())
    else:
        scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar!r}")
    return scalar


def _validate_ig_steps(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("ig_steps must be an integer")
    if value < 2:
        raise ValueError("ig_steps must be at least 2")
    return value


def _validate_internal_batch_size(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("internal_batch_size must be an integer")
    if value <= 0:
        raise ValueError("internal_batch_size must be positive")
    return value


def _validated_sample_id(value) -> str:
    sample_id = str(value)
    if not _SAFE_SAMPLE_ID.fullmatch(sample_id):
        raise ValueError(
            "sample_id must contain only letters, numbers, underscores, or hyphens"
        )
    return sample_id


def _layer_map_for_json(values: dict[int, float]) -> dict[str, float]:
    output = {}
    for layer in sorted(values):
        value = float(values[layer])
        if not math.isfinite(value):
            raise ValueError(f"layer {layer} value must be finite")
        output[str(int(layer))] = value
    return output
