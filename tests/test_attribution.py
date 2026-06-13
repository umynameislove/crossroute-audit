"""Phase-4 attribution tests: CPU-safe contracts and optional BLIP-2 smoke."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torch import nn
from transformers import (
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2QFormerConfig,
    Blip2VisionConfig,
    T5Config,
)

from crossroute_audit.attribution.completeness import (
    completeness_residual,
    convergence_delta_residual,
)
from crossroute_audit.attribution.integrated_gradients import (
    build_attribution_mass_result,
    layer_integrated_gradients_all_layers,
    load_ig_steps,
    write_attribution_mass,
)
from crossroute_audit.attribution.method_agreement import (
    gradient_x_activation_mass,
)
from crossroute_audit.metrics.attribution_mass import (
    attribution_mass_by_layer,
    attribution_mass_for_manifest,
    attribution_mass_for_layer,
)
from crossroute_audit.model_adapters.base import TokenGroups
from crossroute_audit.model_adapters.blip2_adapter import BLIP2Adapter


class TinyBlock(nn.Module):
    """Tuple-returning block that mirrors the relevant T5Block contract."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.projection.weight)

    def forward(self, hidden_states):
        transformed = torch.tanh(self.projection(hidden_states))
        position_bias = torch.zeros(
            hidden_states.shape[0],
            1,
            hidden_states.shape[1],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return transformed, position_bias


class TinyLanguageModel(nn.Module):
    def __init__(self, hidden_size: int = 3, layer_count: int = 2):
        super().__init__()
        self.encoder = SimpleNamespace(
            block=nn.ModuleList(
                [TinyBlock(hidden_size) for _ in range(layer_count)]
            )
        )

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        decoder_input_ids,
        output_hidden_states,
        output_attentions,
        return_dict,
        use_cache,
    ):
        del (
            decoder_input_ids,
            output_hidden_states,
            output_attentions,
            return_dict,
            use_cache,
        )
        hidden_states = inputs_embeds
        for block in self.encoder.block:
            hidden_states = block(hidden_states)[0]
        masked = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = masked.sum(dim=(1, 2))
        logits = torch.stack(
            (
                -pooled,
                pooled,
                0.25 * pooled,
            ),
            dim=-1,
        ).unsqueeze(1)
        return SimpleNamespace(logits=logits)


class TinyAttributionAdapter(BLIP2Adapter):
    def __init__(self):
        super().__init__(device="cpu")
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                use_decoder_only_language_model=False,
                text_config=SimpleNamespace(
                    decoder_start_token_id=0,
                    pad_token_id=0,
                    vocab_size=3,
                ),
            ),
            language_model=TinyLanguageModel(),
        )
        self.processor = SimpleNamespace()
        self._embeddings = torch.tensor(
            [
                [
                    [0.8, 0.2, 0.1],
                    [0.4, 0.3, 0.2],
                    [0.1, 0.5, 0.4],
                    [0.2, 0.6, 0.3],
                ]
            ],
            dtype=torch.float32,
        )

    def prepare_inputs(self, image, question):
        del image, question
        return {
            "encoder_embeddings": self._embeddings.clone(),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }

    def prepare_attribution_inputs(self, inputs, target_answer, policy):
        del target_answer, policy
        return (
            inputs["encoder_embeddings"].detach().requires_grad_(True),
            inputs["attention_mask"],
            1,
        )

    def get_token_groups(self, inputs):
        del inputs
        return TokenGroups(
            image=[0, 1],
            text=[2, 3],
            fusion=[0, 1, 2, 3],
            answer=[0],
        )


def sample(**overrides):
    record = {
        "sample_id": "tiny_001",
        "image_path": "unused.jpg",
        "question": "Is the target present?",
        "target_answer": "yes",
        "target_token_policy": "selected_candidate_logit",
        "expected_visual_dependency": "high",
        "text_only_answerable": "no",
        "counterfactual_image_path": None,
        "control_type": "clean",
        "expected_flip": None,
        "label": None,
        "notes": "Tiny attribution test.",
    }
    record.update(overrides)
    return record


def contains_tensor(value) -> bool:
    if torch.is_tensor(value):
        return True
    if isinstance(value, dict):
        return any(contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_tensor(item) for item in value)
    return False


def test_completeness_residual_is_normalized_and_validated():
    assert completeness_residual(3.0, 5.0, 2.0) == pytest.approx(0.0)
    assert completeness_residual(2.4, 5.0, 2.0) == pytest.approx(0.2)
    assert convergence_delta_residual(-0.3, 5.0, 2.0) == pytest.approx(0.1)

    with pytest.raises(ValueError, match="finite"):
        completeness_residual(float("nan"), 1.0, 0.0)
    with pytest.raises(ValueError, match="positive"):
        completeness_residual(1.0, 1.0, 0.0, epsilon=0.0)


def test_attribution_mass_grouping_and_full_layer_axis():
    groups = TokenGroups(image=[0, 2], text=[1, 3])
    layer_mass = attribution_mass_for_layer(
        [1.0, -2.0, -3.0, 4.0],
        groups,
    )
    assert layer_mass == {"image": 4.0, "text": 6.0}

    result = attribution_mass_by_layer(
        {
            0: [1.0, -2.0, -3.0, 4.0],
            1: [0.5, 0.5, -1.0, -1.0],
        },
        groups,
        layer_count=2,
    )
    assert result == {
        "image": {0: 4.0, 1: 1.5},
        "text": {0: 6.0, 1: 1.5},
    }

    with pytest.raises(ValueError, match="layer axis mismatch"):
        attribution_mass_by_layer({0: [1.0] * 4}, groups, layer_count=2)
    with pytest.raises(IndexError, match="outside"):
        attribution_mass_for_layer([1.0, 2.0], groups)


def test_layer_ig_runs_on_tiny_model_without_hook_leaks():
    adapter = TinyAttributionAdapter()
    inputs = adapter.prepare_inputs(None, "Question?")

    run = layer_integrated_gradients_all_layers(
        adapter,
        inputs,
        "yes",
        "selected_candidate_logit",
        n_steps=8,
        internal_batch_size=2,
    )

    assert set(run.layers) == {0, 1}
    assert run.target_token_id == 1
    assert run.ig_steps == 8
    for layer, result in run.layers.items():
        assert result.layer == layer
        assert len(result.token_attribution) == 4
        assert all(math.isfinite(value) for value in result.token_attribution)
        assert math.isfinite(result.completeness_residual)
        assert math.isfinite(result.convergence_delta)
    assert all(
        len(block._forward_hooks) == 0
        for block in adapter.model.language_model.encoder.block
    )


def test_attribution_artifact_and_secondary_method_share_contract():
    adapter = TinyAttributionAdapter()
    inputs = adapter.prepare_inputs(None, "Question?")

    primary = build_attribution_mass_result(
        adapter,
        inputs,
        sample(),
        n_steps=8,
        internal_batch_size=2,
    )
    secondary = gradient_x_activation_mass(
        adapter,
        inputs,
        "yes",
        "selected_candidate_logit",
    )

    assert set(primary["attribution_mass"]) == {"image", "text"}
    assert set(secondary) == {"image", "text"}
    assert set(primary["attribution_mass"]["image"]) == {"0", "1"}
    assert set(secondary["image"]) == {0, 1}
    assert primary["settings"]["layer_axis"] == "language_model.encoder.block"
    assert primary["settings"]["baseline"] == "zero_lm_encoder_embeddings"
    assert contains_tensor(primary) is False
    assert all(
        math.isfinite(value)
        for group in secondary.values()
        for value in group.values()
    )


def test_attribution_layer_tap_is_removed_when_forward_fails():
    adapter = TinyAttributionAdapter()
    block = adapter.model.language_model.encoder.block[0]

    with pytest.raises(RuntimeError, match="expected"):
        with adapter.attribution_layer_output(0):
            raise RuntimeError("expected")
    assert len(block._forward_hooks) == 0


def test_grad_forward_matches_clean_forward_on_tiny_blip2():
    vision_config = Blip2VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=4,
    )
    qformer_config = Blip2QFormerConfig(
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
    config = Blip2Config.from_vision_qformer_text_configs(
        vision_config,
        qformer_config,
        text_config,
        num_query_tokens=2,
    )
    config.image_token_id = 63

    class TinyTokenizer:
        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return [1]

        def decode(self, token_ids, skip_special_tokens=True):
            del token_ids, skip_special_tokens
            return "yes"

    adapter = BLIP2Adapter(device="cpu")
    adapter.model = Blip2ForConditionalGeneration(config).eval()
    adapter.processor = SimpleNamespace(tokenizer=TinyTokenizer())
    inputs = {
        "pixel_values": torch.randn(1, 3, 8, 8),
        "input_ids": torch.tensor([[63, 63, 5, 6]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }

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

    assert differentiable[0].item() == pytest.approx(clean, abs=1e-6)
    assert gradient.shape == embeddings.shape
    assert torch.isfinite(gradient).all()
    assert len(adapter.model.language_model.encoder._forward_pre_hooks) == 0


def test_manifest_writer_is_deterministic_and_tensor_free(tmp_path):
    adapter = TinyAttributionAdapter()
    manifest_path = tmp_path / "samples.jsonl"
    manifest_path.write_text(json.dumps(sample()) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    paths = attribution_mass_for_manifest(
        adapter,
        manifest_path,
        output_dir,
        n_steps=4,
        internal_batch_size=2,
    )
    first_bytes = (output_dir / "attribution_mass_tiny_001.json").read_bytes()
    paths_again = attribution_mass_for_manifest(
        adapter,
        manifest_path,
        output_dir,
        n_steps=4,
        internal_batch_size=2,
    )
    second_bytes = (output_dir / "attribution_mass_tiny_001.json").read_bytes()
    loaded = json.loads(first_bytes)

    assert paths == paths_again == [
        str(output_dir / "attribution_mass_tiny_001.json")
    ]
    assert first_bytes == second_bytes
    assert loaded["sample_id"] == "tiny_001"
    assert contains_tensor(loaded) is False


def test_artifact_paths_reject_unsafe_ids_and_non_finite_json(tmp_path):
    with pytest.raises(ValueError, match="sample_id"):
        build_attribution_mass_result(
            None,
            None,
            sample(sample_id="../escape"),
            n_steps=2,
        )

    out_path = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        write_attribution_mass({"value": float("nan")}, out_path)
    assert out_path.exists() is False
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_ig_steps_reads_project_config():
    assert load_ig_steps() == 32


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="BLIP-2 attribution smoke test requires CUDA",
)
def test_blip2_gpu_attribution_mass_has_full_layer_axis():
    adapter = BLIP2Adapter()
    inputs = adapter.prepare_inputs(
        Image.new("RGB", (224, 224), color=(40, 100, 220)),
        "Question: is this a blue image? Answer:",
    )

    result = build_attribution_mass_result(
        adapter,
        inputs,
        sample(
            sample_id="gpu_blue",
            target_answer="yes",
            target_token_policy="selected_candidate_logit",
        ),
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
