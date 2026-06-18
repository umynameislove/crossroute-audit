"""Hidden-state ablation in the adapter-defined audit layer space."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from crossroute_audit.instrumentation.hooks import ActivationHook, managed_forward_hook


_ABLATION_BEHAVIORS = {
    "zero": "zero",
    "mean": "mean",
}


def group_positions(adapter, inputs, group: str) -> tuple[int, ...]:
    """Return stable image or text positions from the adapter token groups."""
    if group not in {"image", "text"}:
        raise ValueError("ablation group must be 'image' or 'text'")
    groups = adapter.get_token_groups(inputs)
    positions = tuple(getattr(groups, group))
    if not positions:
        raise ValueError(f"token group {group!r} is empty")
    return positions


def run_ablation(
    adapter,
    inputs,
    layer_module,
    group: str,
    method: str,
    run_forward: Callable[[Any], Any],
) -> Any:
    """Run one zero or mean-mask ablation and return ``run_forward`` output."""
    if method not in _ABLATION_BEHAVIORS:
        supported = ", ".join(sorted(_ABLATION_BEHAVIORS))
        raise ValueError(f"unsupported ablation method {method!r}; expected {supported}")
    hook = ActivationHook(
        behavior=_ABLATION_BEHAVIORS[method],
        positions=group_positions(adapter, inputs, group),
    )
    with managed_forward_hook(layer_module, hook):
        return run_forward(inputs)
