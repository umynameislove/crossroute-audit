"""CPU-safe tests and optional GPU smoke for QwenVLAdapter."""
from __future__ import annotations

import math

from PIL import Image
import pytest
import torch
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

from crossroute_audit.attribution.integrated_gradients import build_attribution_mass_result
from crossroute_audit.model_adapters.base import ModelAdapter
from crossroute_audit.model_adapters.qwenvl_adapter import QwenVLAdapter


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

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

    def convert_tokens_to_ids(self, token):
        if token == "<|image_pad|>":
            return 63
        return 0


def make_tiny_qwenvl_adapter() -> QwenVLAdapter:
    config = Qwen2VLConfig(
        text_config={
            "vocab_size": 80,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "embed_dim": 16,
            "hidden_size": 16,
            "mlp_ratio": 2,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 1,
        },
        image_token_id=63,
    )
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"

    adapter = QwenVLAdapter(device="cpu")
    adapter.model = Qwen2VLForConditionalGeneration(config).eval()
    adapter.processor = type("TinyProcessor", (), {"tokenizer": TinyTokenizer()})()
    return adapter


def tiny_inputs(image_token_count: int = 4) -> dict:
    return {
        "pixel_values": torch.randn(image_token_count, 12),
        "input_ids": torch.tensor(
            [[*[63] * image_token_count, 5, 6]],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(1, image_token_count + 2, dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 1, image_token_count]], dtype=torch.long),
    }


def contains_tensor(value) -> bool:
    if torch.is_tensor(value):
        return True
    if isinstance(value, dict):
        return any(contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_tensor(item) for item in value)
    return False


def test_qwenvl_adapter_conforms_to_model_adapter_contract():
    adapter = QwenVLAdapter(device="cpu")

    assert isinstance(adapter, ModelAdapter)
    assert adapter.layer_axis_name() == "model.model.language_model.layers"
    assert adapter.name == "qwenvl"


def test_tiny_qwenvl_grad_forward_matches_clean_forward_and_groups_image_tokens():
    torch.manual_seed(1234)
    adapter = make_tiny_qwenvl_adapter()
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
    image_token_count = int((inputs["input_ids"] == adapter._image_token_id()).sum().item())

    assert differentiable[0].item() == pytest.approx(clean, abs=1e-4)
    assert gradient.shape == embeddings.shape
    assert torch.isfinite(gradient).all()
    assert len(groups.image) == image_token_count
    assert groups.image == [0, 1, 2, 3]
    assert groups.text == [4, 5]
    assert len(adapter._language_model()._forward_pre_hooks) == 0


def test_tiny_qwenvl_image_group_count_is_dynamic():
    adapter = make_tiny_qwenvl_adapter()
    inputs = tiny_inputs(image_token_count=3)

    groups = adapter.get_token_groups(inputs)

    assert groups.image == [0, 1, 2]
    assert groups.text == [3, 4]
    assert len(groups.image) == int((inputs["input_ids"] == 63).sum().item())


def test_tiny_qwenvl_forward_capture_and_target_policies_are_tensor_free():
    torch.manual_seed(1234)
    adapter = make_tiny_qwenvl_adapter()
    inputs = tiny_inputs()

    output = adapter.forward(inputs, capture=True)

    assert math.isfinite(output.target_logit)
    assert output.hidden_states["decoder"]
    assert output.attentions["decoder"]
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


def test_tiny_qwenvl_builds_attribution_mass_through_shared_pipeline():
    torch.manual_seed(1234)
    adapter = make_tiny_qwenvl_adapter()
    inputs = tiny_inputs()

    result = build_attribution_mass_result(
        adapter,
        inputs,
        {
            "sample_id": "tiny_qwenvl",
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


def test_tiny_qwenvl_attribution_float32_restores_decoder_state_after_failure():
    adapter = make_tiny_qwenvl_adapter()
    adapter.model.model.language_model.half()
    adapter.model.lm_head.half()

    with pytest.raises(RuntimeError, match="expected"):
        with adapter.attribution_float32():
            assert {
                parameter.dtype
                for parameter in adapter.model.model.language_model.parameters()
            } == {torch.float32}
            assert {
                parameter.dtype for parameter in adapter.model.lm_head.parameters()
            } == {torch.float32}
            raise RuntimeError("expected")

    assert {
        parameter.dtype for parameter in adapter.model.model.language_model.parameters()
    } == {torch.float16}
    assert {
        parameter.dtype for parameter in adapter.model.lm_head.parameters()
    } == {torch.float16}


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Qwen2-VL attribution smoke test requires CUDA",
)
def test_qwenvl_gpu_attribution_mass_has_full_layer_axis():
    adapter = QwenVLAdapter()
    inputs = adapter.prepare_inputs(
        Image.new("RGB", (336, 336), color=(40, 100, 220)),
        "is this a blue image?",
    )

    result = build_attribution_mass_result(
        adapter,
        inputs,
        {
            "sample_id": "qwenvl_gpu_blue",
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
