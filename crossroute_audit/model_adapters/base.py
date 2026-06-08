"""Model adapter interface.

The core audit engine is model-agnostic: it never references model-specific
module names. Each adapter exposes token groups, layers, routing proxies,
target logits, and intervention outputs through this contract. BLIP-2 is the
first adapter; LLaVA and Qwen-VL are planned for a later phase.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenGroups:
    """Token indices grouped by role, used to aggregate per-group metrics."""

    image: list[int] = field(default_factory=list)
    text: list[int] = field(default_factory=list)
    fusion: list[int] = field(default_factory=list)
    answer: list[int] = field(default_factory=list)


@dataclass
class ForwardOutput:
    """Result of a forward pass. Tensors are aggregated on capture, never stored raw."""

    target_logit: float
    hidden_states: Any = None
    attentions: Any = None
    meta: dict = field(default_factory=dict)


class ModelAdapter(ABC):
    """Minimal contract every model adapter must implement."""

    name: str = "base"

    @abstractmethod
    def prepare_inputs(self, image, question: str) -> Any:
        """Load and process an image and question into deterministic model inputs."""

    @abstractmethod
    def forward(self, inputs, capture: bool = False) -> ForwardOutput:
        """Run the model. When ``capture`` is true, return attentions and hidden states."""

    @abstractmethod
    def get_token_groups(self, inputs) -> TokenGroups:
        """Return image/text/fusion/answer token groups. Must be stable and logged."""

    @abstractmethod
    def get_layer_count(self) -> int:
        """Return the number of layers exposed for analysis."""

    @abstractmethod
    def get_routing_proxy(self, inputs, layer: int):
        """Return an aggregated attention/interaction proxy for the given layer."""

    @abstractmethod
    def intervene(self, inputs, layer: int, group: str, mode: str):
        """Ablate, mask, patch, or shuffle a route and return the intervened target logit.

        ``mode`` is one of {ablate, mask, patch, shuffle, noop}. The ``noop`` mode is a
        required control: it must leave the clean target logit effectively unchanged.
        """

    @abstractmethod
    def get_target_logit(self, inputs, target_answer: str, policy: str) -> float:
        """Extract the scalar target score according to the target-token policy."""

    @abstractmethod
    def run_controls(self, inputs, sample: dict) -> dict:
        """Run text-only/no-image and counterfactual controls; return control status."""
