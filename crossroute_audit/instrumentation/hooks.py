"""Managed forward hooks for activation capture and intervention.

The hook context managers in this module always remove registered handles,
including when model execution raises. Hook edits are applied to cloned hidden
states so a single intervention cannot mutate tensors retained elsewhere.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


_SUPPORTED_BEHAVIORS = {"capture", "zero", "mean", "replace", "shuffle", "noop"}


def _split_primary_tensor(output: Any) -> tuple[torch.Tensor, Callable[[torch.Tensor], Any]]:
    """Return the primary hidden-state tensor and a function that rebuilds output."""
    if torch.is_tensor(output):
        return output, lambda hidden: hidden
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0], lambda hidden: (hidden, *output[1:])
    if isinstance(output, list) and output and torch.is_tensor(output[0]):
        return output[0], lambda hidden: [hidden, *output[1:]]
    raise TypeError(
        "hooked module output must be a tensor or a sequence whose first item is a tensor"
    )


@dataclass
class ActivationHook:
    """Capture or edit the primary hidden-state tensor from a module output.

    ``positions`` index sequence dimension 1 of a ``[batch, sequence, hidden]``
    activation. ``replace`` accepts either a full activation or a tensor shaped
    ``[batch, len(positions), hidden]``. Captured tensors stay internal and are
    detached clones; callers must aggregate them before serialization.
    """

    behavior: str
    positions: Sequence[int] = ()
    replacement: torch.Tensor | None = None
    captured: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.behavior not in _SUPPORTED_BEHAVIORS:
            supported = ", ".join(sorted(_SUPPORTED_BEHAVIORS))
            raise ValueError(f"unsupported hook behavior {self.behavior!r}; expected {supported}")
        normalized = tuple(int(position) for position in self.positions)
        if any(position < 0 for position in normalized):
            raise ValueError("activation positions must be non-negative")
        if len(set(normalized)) != len(normalized):
            raise ValueError("activation positions must not contain duplicates")
        self.positions = normalized
        if self.behavior in {"zero", "mean", "replace", "shuffle"} and not normalized:
            raise ValueError(f"{self.behavior} requires at least one activation position")
        if self.behavior == "replace" and self.replacement is None:
            raise ValueError("replace requires a replacement tensor")

    def __call__(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> Any:
        hidden, rebuild = _split_primary_tensor(output)

        if self.behavior == "capture":
            self.captured = hidden.detach().clone()
            return None
        if self.behavior == "noop":
            return output
        if hidden.ndim != 3:
            raise ValueError(
                "activation editing expects [batch, sequence, hidden], "
                f"got shape {list(hidden.shape)}"
            )

        index = torch.tensor(self.positions, dtype=torch.long, device=hidden.device)
        if int(index.max().item()) >= hidden.shape[1]:
            raise IndexError(
                f"activation position {int(index.max().item())} is outside "
                f"sequence length {hidden.shape[1]}"
            )

        selected = hidden.index_select(1, index)
        if self.behavior == "zero":
            values = torch.zeros_like(selected)
        elif self.behavior == "mean":
            source_mask = torch.ones(hidden.shape[1], dtype=torch.bool, device=hidden.device)
            source_mask[index] = False
            source = hidden[:, source_mask, :] if bool(source_mask.any()) else hidden
            values = source.mean(dim=1, keepdim=True).expand_as(selected)
        elif self.behavior == "shuffle":
            values = torch.roll(selected, shifts=1, dims=1)
        else:
            values = self._replacement_values(hidden, selected, index)

        edited = hidden.clone()
        edited.index_copy_(1, index, values)
        return rebuild(edited)

    def _replacement_values(
        self,
        hidden: torch.Tensor,
        selected: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        replacement = self.replacement.to(device=hidden.device, dtype=hidden.dtype)
        if replacement.shape == hidden.shape:
            replacement = replacement.index_select(1, index)
        if replacement.shape != selected.shape:
            raise ValueError(
                "replacement must match the full activation or selected positions; "
                f"got {list(replacement.shape)}, expected {list(selected.shape)}"
            )
        return replacement


@contextmanager
def managed_forward_hooks(
    registrations: Iterable[tuple[nn.Module, Callable[..., Any]]],
) -> Iterator[None]:
    """Register forward hooks and guarantee removal in reverse order."""
    handles = []
    try:
        for module, hook in registrations:
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


@contextmanager
def managed_forward_hook(
    module: nn.Module,
    hook: Callable[..., Any],
) -> Iterator[None]:
    """Convenience wrapper for one managed forward hook."""
    with managed_forward_hooks([(module, hook)]):
        yield
