"""Clean-corrupt activation patching in the LM encoder layer space."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from crossroute_audit.instrumentation.hooks import ActivationHook, managed_forward_hook
from crossroute_audit.interventions.ablation import group_positions


def run_activation_patching(
    adapter,
    clean_inputs,
    corrupt_inputs,
    layer_module,
    group: str,
    run_forward: Callable[[Any], Any],
) -> Any:
    """Patch clean group activations into a corrupt run.

    The clean activation is cached only in memory. The returned value is the
    caller's aggregated forward result, never the cached tensor.
    """
    clean_positions = group_positions(adapter, clean_inputs, group)
    corrupt_positions = group_positions(adapter, corrupt_inputs, group)
    if len(clean_positions) != len(corrupt_positions):
        raise ValueError(
            "clean and corrupt token groups must have equal lengths for patching; "
            f"got {len(clean_positions)} and {len(corrupt_positions)}"
        )

    capture = ActivationHook("capture")
    with managed_forward_hook(layer_module, capture):
        run_forward(clean_inputs)
    if capture.captured is None:
        raise RuntimeError("clean activation hook did not capture an activation")

    clean_values = capture.captured[:, clean_positions, :].clone()
    patch = ActivationHook(
        "replace",
        positions=corrupt_positions,
        replacement=clean_values,
    )
    with managed_forward_hook(layer_module, patch):
        return run_forward(corrupt_inputs)
