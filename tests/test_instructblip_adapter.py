"""CPU-safe tests and optional GPU smoke for InstructBLIPAdapter."""
from __future__ import annotations

import math

from PIL import Image
import pytest
import torch
from transformers import (
    InstructBlipConfig,
    InstructBlipForConditionalGeneration,
    InstructBlipQFormerConfig,
    InstructBlipVisionConfig,
    T5Config,
)

from crossroute_audit.attribution.integrated_gradients import build_attribution_mass_result
from crossroute_audit.model_adapters.base import ModelAdapter
from crossroute_audit.model_adapters.instructblip_adapter import InstructBLIPAdapter


class TinyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        mapping = {
            "yes": [7],
            "candidate answer": [8, 9],
            "empty": [],
        }
        return mapping[text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return {0: "<pad>", 7: "yes", 8: "candidate", 9: "answer"}.get(
            int(token_ids[0]),
            "token",
        )


def make_tiny_instructblip_adapter() -> InstructBLIPAdapter:
    vision_config = InstructBlipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=4,
    )
    qformer_config = InstructBlipQFormerConfig(
        vocab_size=64,
        hidden_size=16,
        encoder_hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        cross_attention_frequency=1,
    )
    text_config = T5Config(
        vocab_size=64,
        d_model=16,
        d_ff=32,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        decoder_start_token_id=0,
        pad_token_id=0,
    )
    config = InstructBlipConfig(
        vision_config=vision_config.to_dict(),
        qformer_config=qformer_config.to_dict(),
        text_config=text_config.to_dict(),
        num_query_tokens=2,
        image_token_index=63,
    )
    config._attn_implementation = "eager"
    adapter = InstructBLIPAdapter(device="cpu")
    adapter.model = InstructBlipForConditionalGeneration(config).eval()
    adapter.processor = type("TinyProcessor", (), {"tokenizer": TinyTokenizer()})()
    return adapter


def tiny_inputs() -> dict:
    return {
        "pixel_values": torch.randn(1, 3, 8, 8),
        "qformer_input_ids": torch.tensor([[11, 12, 0]], dtype=torch.long),
        "qformer_attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
        "input_ids": torch.tensor([[63, 63, 5, 6]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }


def contains_tensor(value) -> bool:
    if torch.is_tensor(value):
        return True
    if isinstance(value, dict):
        return any(contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_tensor(item) for item in value)
    return False


def test_instructblip_adapter_conforms_to_model_adapter_contract():
    adapter = InstructBLIPAdapter(device="cpu")

    assert isinstance(adapter, ModelAdapter)
    assert adapter.layer_axis_name() == "language_model.encoder.block"
    assert adapter.name == "instructblip"


def test_tiny_instructblip_grad_forward_matches_clean_forward_and_image_group():
    torch.manual_seed(1234)
    adapter = make_tiny_instructblip_adapter()
    inputs = tiny_inputs()

    clean = adapter.get_target_logit(inputs, "yes", "exact_token")
    embeddings, attention_mask, target_token_id = adapter.prepare_attribution_inputs(
        inputs,
        "yes",
        "exact_token",
    )
    differentiable = adapter.forward_target_logit_from_embeddings(
        embeddings,
        attention_mask,
        target_token_id,
    )
    gradient = torch.autograd.grad(differentiable.sum(), embeddings)[0]
    groups = adapter.get_token_groups(inputs)

    assert differentiable[0].item() == pytest.approx(clean, abs=1e-4)
    assert gradient.shape == embeddings.shape
    assert torch.isfinite(gradient).all()
    assert len(groups.image) == int(adapter.model.config.num_query_tokens)
    assert groups.image == [0, 1]
    assert groups.text == [2, 3]
    assert len(adapter.model.language_model.encoder._forward_pre_hooks) == 0


def test_tiny_instructblip_forward_capture_and_target_policies_are_tensor_free():
    torch.manual_seed(1234)
    adapter = make_tiny_instructblip_adapter()
    inputs = tiny_inputs()

    output = adapter.forward(inputs, capture=True)

    assert math.isfinite(output.target_logit)
    assert output.hidden_states["language_encoder"]
    assert output.attentions["language_encoder"]
    assert contains_tensor(output.__dict__) is False
    assert adapter.get_target_logit(
        inputs,
        "candidate answer",
        "selected_candidate_logit",
    ) == pytest.approx(
        adapter.get_target_logit(inputs, "candidate answer", "selected_candidate_logit")
    )
    with pytest.raises(ValueError, match="exact_token"):
        adapter.get_target_logit(inputs, "candidate answer", "exact_token")


def test_tiny_instructblip_builds_attribution_mass_through_shared_pipeline():
    torch.manual_seed(1234)
    adapter = make_tiny_instructblip_adapter()
    inputs = tiny_inputs()

    result = build_attribution_mass_result(
        adapter,
        inputs,
        {
            "sample_id": "tiny_instructblip",
            "target_answer": "yes",
            "target_token_policy": "exact_token",
        },
        n_steps=2,
        internal_batch_size=1,
    )

    assert result["settings"]["layer_axis"] == adapter.layer_axis_name()
    assert set(result["attribution_mass"]) == {"image", "text"}
    assert set(result["attribution_mass"]["image"]) == {"0", "1"}
    assert set(result["attribution_mass"]["text"]) == {"0", "1"}
    assert contains_tensor(result) is False
    assert all(
        math.isfinite(value)
        for group in result["attribution_mass"].values()
        for value in group.values()
    )


def test_tiny_instructblip_attribution_layer_tap_is_removed_when_forward_fails():
    adapter = make_tiny_instructblip_adapter()
    layer = adapter._lm_encoder_layers()[0]

    with pytest.raises(RuntimeError, match="expected"):
        with adapter.attribution_layer_output(0):
            raise RuntimeError("expected")

    assert len(layer._forward_hooks) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="InstructBLIP attribution smoke test requires CUDA",
)
def test_instructblip_gpu_attribution_mass_has_full_layer_axis():
    adapter = InstructBLIPAdapter()
    inputs = adapter.prepare_inputs(
        Image.new("RGB", (224, 224), color=(40, 100, 220)),
        "is this a blue image?",
    )

    result = build_attribution_mass_result(
        adapter,
        inputs,
        {
            "sample_id": "instructblip_gpu_blue",
            "target_answer": "yes",
            "target_token_policy": "selected_candidate_logit",
        },
        n_steps=2,
        internal_batch_size=1,
    )

    expected_layers = {
        str(layer) for layer in range(adapter.get_intervention_layer_count())
    }
    assert set(result["attribution_mass"]) == {"image", "text"}
    assert set(result["attribution_mass"]["image"]) == expected_layers
    assert set(result["attribution_mass"]["text"]) == expected_layers
    assert contains_tensor(result) is False
    assert all(
        math.isfinite(value)
        for group in result["attribution_mass"].values()
        for value in group.values()
    )
