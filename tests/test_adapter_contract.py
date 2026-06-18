"""Architecture-agnostic ModelAdapter contract tests."""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from crossroute_audit.attribution.integrated_gradients import (
    LayerIGAttribution,
    LayerIGRun,
    build_attribution_mass_result,
)
from crossroute_audit.model_adapters.base import ForwardOutput, ModelAdapter, TokenGroups
from crossroute_audit.model_adapters.blip2_adapter import BLIP2Adapter


class BaseContractAdapter(ModelAdapter):
    name = "contract"

    def prepare_inputs(self, image, question: str) -> dict:
        return {"image": image, "question": question}

    def forward(self, inputs, capture: bool = False) -> ForwardOutput:
        return ForwardOutput(target_logit=1.0)

    def get_token_groups(self, inputs) -> TokenGroups:
        return TokenGroups(image=[0], text=[1], fusion=[0, 1], answer=[0])

    def get_layer_count(self) -> int:
        return 1

    def get_intervention_layer_count(self) -> int:
        return 1

    def get_routing_proxy(self, inputs, layer: int):
        return 0.0

    def intervene(self, inputs, layer: int, group: str, mode: str):
        return 1.0

    def get_target_logit(self, inputs, target_answer: str, policy: str) -> float:
        return 1.0

    def prepare_attribution_inputs(self, inputs, target_answer: str, policy: str):
        raise NotImplementedError

    def attribution_baseline_embeddings(self, inputs):
        raise NotImplementedError

    def forward_target_logit_from_embeddings(
        self,
        embeddings,
        attention_mask,
        target_token_id: int,
    ):
        raise NotImplementedError

    @contextmanager
    def attribution_layer_output(self, layer: int):
        yield None

    @contextmanager
    def attribution_float32(self):
        yield

    def run_controls(self, inputs, sample: dict) -> dict:
        return {}


class CompleteContractAdapter(BaseContractAdapter):
    def layer_axis_name(self) -> str:
        return "contract.audit_layers"


def test_model_adapter_declares_full_contract_surface():
    expected = {
        "prepare_inputs",
        "forward",
        "get_token_groups",
        "get_layer_count",
        "get_intervention_layer_count",
        "layer_axis_name",
        "prepare_attribution_inputs",
        "attribution_baseline_embeddings",
        "forward_target_logit_from_embeddings",
        "attribution_layer_output",
        "attribution_float32",
        "intervene",
        "get_target_logit",
        "get_routing_proxy",
        "run_controls",
    }

    assert expected.issubset(ModelAdapter.__abstractmethods__)


def test_adapter_missing_contract_method_cannot_instantiate():
    with pytest.raises(TypeError, match="layer_axis_name"):
        BaseContractAdapter()


def test_layer_axis_name_is_non_empty_string():
    assert CompleteContractAdapter().layer_axis_name()
    assert BLIP2Adapter(device="cpu").layer_axis_name()


def test_attribution_artifact_uses_adapter_layer_axis(monkeypatch):
    adapter = CompleteContractAdapter()

    def fake_layer_ig_all_layers(
        adapter,
        inputs,
        target_answer: str,
        policy: str,
        *,
        n_steps=None,
        internal_batch_size=1,
    ):
        del inputs, target_answer, policy, n_steps, internal_batch_size
        return LayerIGRun(
            layers={
                0: LayerIGAttribution(
                    layer=0,
                    token_attribution=(1.0, -2.0),
                    completeness_residual=0.0,
                    convergence_delta=0.0,
                )
            },
            target_token_id=1,
            target_logit=3.0,
            baseline_logit=0.5,
            ig_steps=8,
        )

    monkeypatch.setattr(
        "crossroute_audit.attribution.integrated_gradients."
        "layer_integrated_gradients_all_layers",
        fake_layer_ig_all_layers,
    )

    result = build_attribution_mass_result(
        adapter,
        adapter.prepare_inputs("image.jpg", "question?"),
        {
            "sample_id": "contract_001",
            "target_answer": "yes",
            "target_token_policy": "exact_token",
        },
        n_steps=8,
    )

    assert result["settings"]["layer_axis"] == adapter.layer_axis_name()
