"""CPU-safe contract tests and an optional GPU smoke test for BLIP2Adapter."""
from __future__ import annotations

from types import SimpleNamespace

from PIL import Image
import pytest

from crossroute_audit.model_adapters import blip2_adapter as blip2_module
from crossroute_audit.model_adapters.blip2_adapter import BLIP2Adapter


torch = blip2_module.torch


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        mapping = {
            "yes": [2],
            "candidate answer": [1, 3],
            "empty": [],
        }
        return mapping[text]

    def decode(self, token_ids, skip_special_tokens=True):
        return {0: "no", 1: "candidate", 2: "yes", 3: "answer"}[token_ids[0]]


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.num_query_tokens = None
        self.last_image = None
        self.last_text = None

    def __call__(self, images, text, return_tensors):
        self.last_image = images
        self.last_text = text
        return {
            "pixel_values": torch.ones(1, 3, 2, 2),
            "input_ids": torch.tensor([[99, 99, 7, 8, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long),
        }


class FakeConfig:
    def __init__(self):
        self.output_hidden_states = False
        self.output_attentions = False


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(
            num_query_tokens=2,
            use_decoder_only_language_model=False,
            qformer_config=SimpleNamespace(num_hidden_layers=12),
            text_config=SimpleNamespace(
                decoder_start_token_id=0,
                pad_token_id=0,
            ),
            output_hidden_states=False,
            output_attentions=False,
        )
        self.vision_model = SimpleNamespace(config=FakeConfig())
        self.qformer = SimpleNamespace(config=FakeConfig())
        self.language_model = SimpleNamespace(config=FakeConfig())
        self.last_kwargs = None

    def parameters(self):
        yield torch.ones(1, dtype=torch.float16)

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        capture = kwargs["output_hidden_states"]
        hidden = (torch.ones(1, 2, 3),) if capture else None
        attention = (torch.full((1, 1, 2, 2), 0.5),) if capture else None
        language_outputs = SimpleNamespace(
            encoder_hidden_states=hidden,
            decoder_hidden_states=hidden,
            hidden_states=None,
            encoder_attentions=attention,
            decoder_attentions=attention,
            attentions=None,
            cross_attentions=attention,
        )
        return SimpleNamespace(
            logits=torch.tensor([[[0.1, 1.5, 3.0, -0.5]]]),
            vision_outputs=SimpleNamespace(hidden_states=hidden, attentions=attention),
            qformer_outputs=SimpleNamespace(
                hidden_states=hidden,
                attentions=attention,
                cross_attentions=attention,
            ),
            language_model_outputs=language_outputs,
        )


def make_adapter() -> BLIP2Adapter:
    adapter = BLIP2Adapter(device="cpu")
    adapter.processor = FakeProcessor()
    adapter.model = FakeModel()
    return adapter


def contains_tensor(value) -> bool:
    if torch.is_tensor(value):
        return True
    if isinstance(value, dict):
        return any(contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_tensor(item) for item in value)
    return False


def test_model_loading_is_lazy_and_uses_required_settings(monkeypatch):
    processor = FakeProcessor()

    class LoadableFakeModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.to_device = None
            self.eval_called = False

        def to(self, device):
            self.to_device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    model = LoadableFakeModel()
    recorded = {}

    def processor_from_pretrained(model_name):
        recorded["processor_model_name"] = model_name
        return processor

    def model_from_pretrained(model_name, **kwargs):
        recorded["model_name"] = model_name
        recorded["kwargs"] = kwargs
        return model

    monkeypatch.setattr(
        blip2_module.Blip2Processor,
        "from_pretrained",
        processor_from_pretrained,
    )
    monkeypatch.setattr(
        blip2_module.Blip2ForConditionalGeneration,
        "from_pretrained",
        model_from_pretrained,
    )

    adapter = BLIP2Adapter(device="cpu")
    assert adapter.model is None
    adapter._ensure_loaded()

    assert recorded["processor_model_name"] == "Salesforce/blip2-flan-t5-xl"
    assert recorded["model_name"] == "Salesforce/blip2-flan-t5-xl"
    assert recorded["kwargs"]["attn_implementation"] == "eager"
    assert recorded["kwargs"]["torch_dtype"] is torch.float16
    assert processor.num_query_tokens == 2
    assert model.to_device == "cpu"
    assert model.eval_called is True


def test_prepare_inputs_accepts_pil_and_path(tmp_path):
    adapter = make_adapter()
    source = Image.new("L", (3, 2), color=128)
    image_path = tmp_path / "sample.png"
    source.save(image_path)

    from_pil = adapter.prepare_inputs(source, " Is there an object? ")
    assert adapter.processor.last_image.mode == "RGB"
    assert adapter.processor.last_text == "Question: Is there an object? Answer:"
    assert from_pil["pixel_values"].dtype is torch.float16

    from_path = adapter.prepare_inputs(image_path, "What is shown?")
    assert adapter.processor.last_image.mode == "RGB"
    assert from_path["input_ids"].dtype is torch.long


def test_forward_capture_returns_summaries_without_raw_tensors():
    adapter = make_adapter()
    inputs = adapter.prepare_inputs(Image.new("RGB", (2, 2)), "Question?")

    result = adapter.forward(inputs, capture=True)

    assert result.target_logit == pytest.approx(3.0)
    assert result.hidden_states["vision"][0]["shape"] == [1, 2, 3]
    assert result.attentions["qformer_cross"][0]["shape"] == [1, 1, 2, 2]
    assert result.meta["predicted_token_id"] == 2
    assert result.meta["predicted_token"] == "yes"
    assert contains_tensor(result.hidden_states) is False
    assert contains_tensor(result.attentions) is False
    assert adapter.model.last_kwargs["decoder_input_ids"].shape == (1, 1)


def test_target_logit_policies_use_first_decode_step():
    adapter = make_adapter()
    inputs = adapter.prepare_inputs(Image.new("RGB", (2, 2)), "Question?")

    assert adapter.get_target_logit(inputs, "yes", "exact_token") == pytest.approx(3.0)
    assert adapter.get_target_logit(
        inputs,
        "ignored",
        "first_generated_token",
    ) == pytest.approx(3.0)
    assert adapter.get_target_logit(
        inputs,
        "candidate answer",
        "selected_candidate_logit",
    ) == pytest.approx(1.5)

    with pytest.raises(ValueError, match="exactly one token"):
        adapter.get_target_logit(inputs, "candidate answer", "exact_token")


def test_get_layer_count_and_token_groups_are_qformer_based():
    adapter = make_adapter()
    inputs = adapter.prepare_inputs(Image.new("RGB", (2, 2)), "Question?")

    assert adapter.get_layer_count() == 12
    assert adapter.get_token_groups(inputs) == blip2_module.TokenGroups(
        image=[0, 1],
        text=[2, 3],
        fusion=[0, 1, 2, 3],
        answer=[0],
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="BLIP-2 GPU smoke test requires CUDA",
)
def test_blip2_gpu_smoke_forward_and_target_logit():
    adapter = BLIP2Adapter()
    inputs = adapter.prepare_inputs(
        Image.new("RGB", (224, 224), color=(120, 160, 200)),
        "is this a blue image?",
    )

    result = adapter.forward(inputs, capture=True)
    target_logit = adapter.get_target_logit(
        inputs,
        "yes",
        "selected_candidate_logit",
    )

    assert result.hidden_states is not None
    assert result.attentions is not None
    assert torch.isfinite(torch.tensor(target_logit))
