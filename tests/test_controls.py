from __future__ import annotations

import pytest

from crossroute_audit.controls.baselines import (
    run_counterfactual,
    run_controls,
    run_negative_control,
    run_no_image,
    run_text_only,
)
from crossroute_audit.model_adapters.base import ForwardOutput, ModelAdapter, TokenGroups


class MockAdapter(ModelAdapter):
    name = "mock"

    def __init__(self, logits: dict[str, float], negative_control_effect: float = 0.0):
        self.logits = logits
        self.negative_control_effect = negative_control_effect

    def prepare_inputs(self, image, question: str):
        return {"image": image, "question": question}

    def forward(self, inputs, capture: bool = False) -> ForwardOutput:
        return ForwardOutput(target_logit=self._logit_for_image(inputs["image"]))

    def get_token_groups(self, inputs) -> TokenGroups:
        return TokenGroups(image=[0], text=[1], fusion=[2], answer=[3])

    def get_layer_count(self) -> int:
        return 1

    def get_routing_proxy(self, inputs, layer: int):
        return {"layer": layer, "proxy": 1.0}

    def intervene(self, inputs, layer: int, group: str, mode: str):
        clean = self._logit_for_image(inputs["image"])
        return clean - self.negative_control_effect

    def get_target_logit(self, inputs, target_answer: str, policy: str) -> float:
        return self._logit_for_image(inputs["image"])

    def run_controls(self, inputs, sample: dict) -> dict:
        return run_controls(self, sample)

    def _logit_for_image(self, image) -> float:
        if image is None:
            key = "text_only"
        elif isinstance(image, dict) and image.get("kind") == "blank_image":
            key = "blank"
        elif hasattr(image, "getpixel"):
            key = "blank"
        else:
            key = str(image)
        return self.logits[key]


def sample(**overrides) -> dict:
    base = {
        "sample_id": "sample",
        "image_path": "clean.jpg",
        "question": "Is the target present?",
        "target_answer": "yes",
        "target_token_policy": "first_generated_token",
        "expected_visual_dependency": "high",
        "text_only_answerable": "no",
        "control_type": "clean",
        "counterfactual_image_path": None,
        "expected_flip": None,
    }
    base.update(overrides)
    return base


def test_text_only_marks_language_prior_answerable():
    adapter = MockAdapter({"clean.jpg": 3.0, "text_only": 2.8})
    result = run_text_only(adapter, sample(text_only_answerable="yes"))

    assert result == {"target_logit": 2.8, "answerable": "yes"}


def test_text_only_marks_visual_sample_not_answerable():
    adapter = MockAdapter({"clean.jpg": 3.0, "text_only": 0.2})
    result = run_text_only(adapter, sample())

    assert result == {"target_logit": 0.2, "answerable": "no"}


def test_no_image_uses_blank_image_and_marks_prior_answerable():
    adapter = MockAdapter({"clean.jpg": 3.0, "blank": 2.7})
    result = run_no_image(adapter, sample())

    assert result == {"target_logit": 2.7, "answerable": "yes"}


def test_counterfactual_reports_flip_when_delta_is_large():
    adapter = MockAdapter({"clean.jpg": 4.0, "counterfactual.jpg": 2.5})
    result = run_counterfactual(
        adapter,
        sample(counterfactual_image_path="counterfactual.jpg", expected_flip=True),
    )

    assert result["delta_logit_cf"] == pytest.approx(1.5)
    assert result["flipped"] is True
    assert result["expected_flip"] is True


def test_negative_control_effect_is_near_zero():
    adapter = MockAdapter({"clean.jpg": 3.0}, negative_control_effect=0.02)
    result = run_negative_control(adapter, sample(control_type="negative_control"))

    assert result["effect"] == pytest.approx(0.02)


def test_run_controls_can_select_subset():
    adapter = MockAdapter(
        {
            "clean.jpg": 4.0,
            "text_only": 0.1,
            "counterfactual.jpg": 2.8,
        },
        negative_control_effect=0.0,
    )
    result = run_controls(
        adapter,
        sample(counterfactual_image_path="counterfactual.jpg", expected_flip=True),
        which="text_only,counterfactual",
    )

    assert set(result) == {"text_only", "counterfactual"}
    assert result["text_only"]["answerable"] == "no"
    assert result["counterfactual"]["flipped"] is True
