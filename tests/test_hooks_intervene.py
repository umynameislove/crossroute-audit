"""Hook and causal-intervention tests for the BLIP-2 adapter."""
from __future__ import annotations

from types import SimpleNamespace

from PIL import Image
import pytest

from crossroute_audit.instrumentation import hooks as hooks_module
from crossroute_audit.instrumentation.hooks import (
    ActivationHook,
    managed_forward_hook,
    managed_forward_hooks,
)
from crossroute_audit.interventions.ablation import group_positions, run_ablation
from crossroute_audit.interventions.patching import run_activation_patching
from crossroute_audit.model_adapters import blip2_adapter as blip2_module
from crossroute_audit.model_adapters.blip2_adapter import BLIP2Adapter


torch = hooks_module.torch
nn = hooks_module.nn


class TupleBlock(nn.Module):
    def forward(self, hidden):
        return (hidden + 1.0, "metadata")


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"yes": [1], "no": [2]}[text]

    def decode(self, token_ids, skip_special_tokens=True):
        return {0: "other", 1: "yes", 2: "no"}[token_ids[0]]


class FakeInterventionAdapter(BLIP2Adapter):
    def __init__(self):
        super().__init__(device="cpu")
        blocks = nn.ModuleList([TupleBlock(), TupleBlock()])
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                num_query_tokens=2,
                image_token_id=99,
                qformer_config=SimpleNamespace(num_hidden_layers=3),
                text_config=SimpleNamespace(
                    decoder_start_token_id=0,
                    pad_token_id=0,
                ),
                use_decoder_only_language_model=False,
            ),
            language_model=SimpleNamespace(
                encoder=SimpleNamespace(block=blocks),
            ),
        )
        self.processor = SimpleNamespace(tokenizer=FakeTokenizer())

    def _run_first_decode_step(self, inputs, capture: bool):
        hidden = inputs["hidden_states"]
        sequence_length = inputs["input_ids"].shape[1]
        if hidden.shape[1] < sequence_length:
            padding = torch.zeros(
                hidden.shape[0],
                sequence_length - hidden.shape[1],
                hidden.shape[2],
                dtype=hidden.dtype,
            )
            hidden = torch.cat([hidden, padding], dim=1)
        attentions = []
        for block in self._lm_encoder_layers():
            hidden = block(hidden)[0]
            sequence_length = hidden.shape[1]
            attention = torch.zeros(1, 1, sequence_length, sequence_length)
            attention[:, :, :2, 2:4] = 0.25
            attention[:, :, 2:4, :2] = 0.5
            attentions.append(attention)

        image_score = hidden[:, :2, :].sum()
        text_score = hidden[:, 2:4, :].sum()
        logits = torch.stack(
            [
                torch.tensor(0.0),
                image_score + 0.1 * text_score,
                text_score,
            ]
        ).reshape(1, 3)
        outputs = SimpleNamespace(
            language_model_outputs=SimpleNamespace(
                encoder_attentions=tuple(attentions) if capture else None,
            )
        )
        return outputs, logits


def fake_inputs():
    return {
        "input_ids": torch.tensor([[99, 99, 5, 6, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long),
        "hidden_states": torch.tensor(
            [
                [
                    [4.0, 4.0],
                    [3.0, 3.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [9.0, 9.0],
                ]
            ]
        ),
    }


def test_capture_hook_records_clone_and_is_removed():
    block = TupleBlock()
    hook = ActivationHook("capture")
    source = torch.ones(1, 3, 2)

    assert len(block._forward_hooks) == 0
    with managed_forward_hook(block, hook):
        assert len(block._forward_hooks) == 1
        output = block(source)
    assert len(block._forward_hooks) == 0
    assert output[1] == "metadata"
    assert torch.equal(hook.captured, source + 1.0)
    assert hook.captured.data_ptr() != output[0].data_ptr()


def test_hook_is_removed_when_forward_raises():
    class FailingBlock(nn.Module):
        def forward(self, hidden):
            raise RuntimeError("expected failure")

    block = FailingBlock()
    with pytest.raises(RuntimeError, match="expected failure"):
        with managed_forward_hook(block, ActivationHook("noop")):
            block(torch.ones(1, 2, 2))
    assert len(block._forward_hooks) == 0


def test_multiple_hooks_are_removed_when_an_edit_raises():
    first = TupleBlock()
    second = TupleBlock()
    registrations = [
        (first, ActivationHook("noop")),
        (second, ActivationHook("zero", positions=[99])),
    ]

    with pytest.raises(IndexError, match="outside sequence length"):
        with managed_forward_hooks(registrations):
            first(torch.ones(1, 2, 2))
            second(torch.ones(1, 2, 2))
    assert len(first._forward_hooks) == 0
    assert len(second._forward_hooks) == 0


def test_activation_edit_behaviors_preserve_output_structure():
    block = TupleBlock()
    source = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])

    with managed_forward_hook(block, ActivationHook("zero", positions=[0, 1])):
        zeroed = block(source)
    assert zeroed[1] == "metadata"
    assert zeroed[0][0, :2, :].tolist() == [[0.0], [0.0]]

    with managed_forward_hook(block, ActivationHook("mean", positions=[0, 1])):
        masked = block(source)[0]
    assert masked[0, :2, :].tolist() == [[4.5], [4.5]]

    with managed_forward_hook(block, ActivationHook("shuffle", positions=[0, 1, 2])):
        shuffled = block(source)[0]
    assert shuffled[0, :3, :].tolist() == [[4.0], [2.0], [3.0]]

    with managed_forward_hook(block, ActivationHook("noop")):
        unchanged = block(source)[0]
    assert torch.equal(unchanged, source + 1.0)


def test_group_positions_and_ablation_use_adapter_token_groups():
    adapter = FakeInterventionAdapter()
    inputs = fake_inputs()
    layer = adapter._lm_encoder_layers()[0]

    assert group_positions(adapter, inputs, "image") == (0, 1)
    assert group_positions(adapter, inputs, "text") == (2, 3)
    with pytest.raises(ValueError, match="image.*text"):
        group_positions(adapter, inputs, "fusion")

    clean = adapter._run_first_decode_step(inputs, capture=False)[1][0, 1].item()
    ablated = run_ablation(
        adapter,
        inputs,
        layer,
        "image",
        "zero",
        lambda run_inputs: adapter._run_first_decode_step(run_inputs, capture=False)[1][
            0, 1
        ].item(),
    )
    assert ablated < clean
    assert len(layer._forward_hooks) == 0


def test_clean_corrupt_patching_restores_group_activation():
    adapter = FakeInterventionAdapter()
    clean_inputs = fake_inputs()
    corrupt_inputs = fake_inputs()
    corrupt_inputs["hidden_states"][:, :2, :] = 0.0
    layer = adapter._lm_encoder_layers()[1]

    def target_logit(run_inputs):
        return adapter._run_first_decode_step(run_inputs, capture=False)[1][0, 1].item()

    corrupt = target_logit(corrupt_inputs)
    patched = run_activation_patching(
        adapter,
        clean_inputs,
        corrupt_inputs,
        layer,
        "image",
        target_logit,
    )
    assert patched > corrupt
    assert len(layer._forward_hooks) == 0


def test_routing_proxy_and_intervention_validation_are_lm_encoder_based():
    adapter = FakeInterventionAdapter()
    inputs = fake_inputs()

    assert adapter.get_layer_count() == 3
    assert adapter.get_intervention_layer_count() == 2
    assert adapter.get_routing_proxy(inputs, layer=1) == pytest.approx(0.75)
    with pytest.raises(IndexError, match="LM encoder layer"):
        adapter.get_routing_proxy(inputs, layer=2)
    with pytest.raises(ValueError, match="intervention mode"):
        adapter.intervene(inputs, layer=0, group="image", mode="unknown")
    with pytest.raises(ValueError, match="intervention group"):
        adapter.intervene(inputs, layer=0, group="fusion", mode="ablate")


def test_noop_isolation_negative_control_and_target_metadata_on_fake_model():
    adapter = FakeInterventionAdapter()
    inputs = fake_inputs()
    clean = adapter.get_target_logit(inputs, "yes", "exact_token")

    noop = adapter.intervene(inputs, layer=1, group="image", mode="noop")
    ablated = adapter.intervene(inputs, layer=1, group="image", mode="ablate")
    clean_after = adapter.get_target_logit(inputs, "yes", "exact_token")
    negative = adapter.intervene(
        inputs,
        layer=1,
        group="negative_control",
        mode="ablate",
    )
    legacy_negative = adapter.intervene(
        inputs,
        layer=1,
        group="image",
        mode="negative_control",
    )

    assert noop == pytest.approx(clean)
    assert ablated < clean
    assert clean_after == pytest.approx(clean)
    assert negative == pytest.approx(clean)
    assert legacy_negative == pytest.approx(clean)
    assert all(
        len(layer._forward_hooks) == 0
        for layer in adapter._lm_encoder_layers()
    )


def test_adapter_patch_mode_uses_clean_activation_on_corrupt_run():
    adapter = FakeInterventionAdapter()
    clean_inputs = fake_inputs()
    corrupt_inputs = fake_inputs()
    corrupt_inputs["hidden_states"][:, :2, :] = 0.0
    clean_inputs["_crossroute_corrupt_inputs"] = corrupt_inputs
    clean = adapter.get_target_logit(clean_inputs, "yes", "exact_token")
    corrupt = adapter.get_target_logit(corrupt_inputs, "yes", "exact_token")

    patched = adapter.intervene(
        clean_inputs,
        layer=1,
        group="image",
        mode="patch",
    )

    assert corrupt < patched <= clean
    assert all(
        len(layer._forward_hooks) == 0
        for layer in adapter._lm_encoder_layers()
    )


@pytest.fixture(scope="module")
def gpu_case():
    if not torch.cuda.is_available():
        pytest.skip("BLIP-2 causal smoke tests require CUDA")
    adapter = BLIP2Adapter()
    inputs = adapter.prepare_inputs(
        Image.new("RGB", (224, 224), color=(40, 100, 220)),
        "is this a blue image?",
    )
    clean = adapter.get_target_logit(inputs, "yes", "selected_candidate_logit")
    layer = len(adapter._lm_encoder_layers()) - 1
    return adapter, inputs, clean, layer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_noop_preserves_target_logit(gpu_case):
    adapter, inputs, clean, layer = gpu_case
    noop = adapter.intervene(inputs, layer=layer, group="image", mode="noop")
    assert noop == pytest.approx(clean, abs=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_intervention_isolation(gpu_case):
    adapter, inputs, clean, layer = gpu_case
    adapter.intervene(inputs, layer=layer, group="image", mode="mask")
    clean_after = adapter.get_target_logit(inputs, "yes", "selected_candidate_logit")
    assert clean_after == pytest.approx(clean, abs=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_image_ablation_reduces_visual_target(gpu_case):
    adapter, inputs, clean, layer = gpu_case
    ablated = adapter.intervene(inputs, layer=layer, group="image", mode="ablate")
    assert ablated < clean


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_negative_control_has_near_zero_effect(gpu_case):
    adapter, inputs, clean, layer = gpu_case
    negative = adapter.intervene(
        inputs,
        layer=layer,
        group="negative_control",
        mode="ablate",
    )
    # FP16 logits have an approximately 2e-3 ULP at this magnitude.
    assert abs(clean - negative) < 5e-3
