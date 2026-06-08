"""BLIP-2 adapter (first supported model).

BLIP-2 exposes a Q-Former with explicit cross-attention, which gives a clear
intervention surface. This module currently defines the adapter shape; the
model-specific logic is implemented incrementally.
"""
from __future__ import annotations

from .base import ForwardOutput, ModelAdapter, TokenGroups


class BLIP2Adapter(ModelAdapter):
    name = "blip2"

    def __init__(self, model_name: str = "Salesforce/blip2-flan-t5-xl", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        # Model and processor are loaded lazily in a later milestone.

    def prepare_inputs(self, image, question):
        raise NotImplementedError

    def forward(self, inputs, capture: bool = False) -> ForwardOutput:
        raise NotImplementedError

    def get_token_groups(self, inputs) -> TokenGroups:
        raise NotImplementedError

    def get_layer_count(self) -> int:
        raise NotImplementedError

    def get_routing_proxy(self, inputs, layer: int):
        raise NotImplementedError

    def intervene(self, inputs, layer: int, group: str, mode: str):
        raise NotImplementedError

    def get_target_logit(self, inputs, target_answer: str, policy: str) -> float:
        raise NotImplementedError

    def run_controls(self, inputs, sample: dict) -> dict:
        raise NotImplementedError
