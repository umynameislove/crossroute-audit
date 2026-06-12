from __future__ import annotations

import json

from jsonschema import Draft202012Validator
import pytest

from crossroute_audit.io.causal_effect import (
    build_causal_effect_result,
    causal_effect_for_manifest,
    write_causal_effect,
)
from crossroute_audit.metrics.causal_effect import (
    causal_effect,
    causal_effect_by_layer,
    effect_stability,
)


class MockAdapter:
    def __init__(self, clean_logit: float = 3.0, layer_count: int = 4):
        self.clean_logit = clean_logit
        self.layer_count = layer_count
        self.effects = {
            "image": [0.2, 0.4, 0.6, 0.8],
            "text": [1.0, 1.2, 1.4, 1.6],
        }
        self.target_calls = []
        self.intervention_calls = []
        self.prepared_images = []

    def prepare_inputs(self, image, question: str):
        self.prepared_images.append(image)
        return {"image": image, "question": question}

    def get_target_logit(self, inputs, target_answer: str, policy: str) -> float:
        self.target_calls.append((inputs["image"], target_answer, policy))
        inputs["_target_resolved"] = (target_answer, policy)
        return self.clean_logit

    def get_intervention_layer_count(self) -> int:
        return self.layer_count

    def intervene(self, inputs, layer: int, group: str, mode: str):
        assert "_target_resolved" in inputs
        self.intervention_calls.append((layer, group, mode))
        return self.clean_logit - self.effects[group][layer]


def sample(**overrides) -> dict:
    base = {
        "sample_id": "sample_001",
        "image_path": "clean.jpg",
        "question": "Is the target present?",
        "target_answer": "yes",
        "target_token_policy": "first_generated_token",
        "expected_visual_dependency": "high",
        "text_only_answerable": "no",
        "counterfactual_image_path": None,
        "control_type": "clean",
        "expected_flip": None,
        "label": None,
        "notes": "Mock sample for causal-effect tests.",
    }
    base.update(overrides)
    return base


def load_causal_effect_schema() -> dict:
    with open("schemas/causal_effect.schema.json", encoding="utf-8") as schema_file:
        return json.load(schema_file)


def test_causal_effect_subtracts_intervened_logit():
    assert causal_effect(3.0, 1.0) == pytest.approx(2.0)


def test_causal_effect_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        causal_effect(float("nan"), 1.0)


def test_causal_effect_by_layer_uses_intervention_layer_axis_and_target_first():
    adapter = MockAdapter(clean_logit=3.0, layer_count=4)
    inputs = adapter.prepare_inputs("clean.jpg", "Question?")

    result = causal_effect_by_layer(adapter, inputs, sample(), group="image")

    assert result == pytest.approx({0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8})
    assert adapter.target_calls == [("clean.jpg", "yes", "first_generated_token")]
    assert adapter.intervention_calls == [
        (0, "image", "ablate"),
        (1, "image", "ablate"),
        (2, "image", "ablate"),
        (3, "image", "ablate"),
    ]


def test_causal_effect_by_layer_requires_intervention_layer_count():
    class BadAdapter:
        pass

    with pytest.raises(AttributeError, match="get_intervention_layer_count"):
        causal_effect_by_layer(BadAdapter(), {}, sample(), group="image")


def test_effect_stability_summarizes_repeated_effects():
    adapter = MockAdapter(clean_logit=3.0, layer_count=4)
    inputs = adapter.prepare_inputs("clean.jpg", "Question?")

    result = effect_stability(
        adapter,
        inputs,
        sample(),
        layer=2,
        group="text",
        repeats=3,
    )

    assert result == pytest.approx({"mean": 1.4, "std": 0.0, "max_abs_dev": 0.0})
    assert len(adapter.target_calls) == 3
    assert adapter.intervention_calls == [
        (2, "text", "ablate"),
        (2, "text", "ablate"),
        (2, "text", "ablate"),
    ]


def test_effect_stability_rejects_invalid_repeats():
    adapter = MockAdapter()

    with pytest.raises(TypeError, match="integer"):
        effect_stability(adapter, {}, sample(), layer=0, group="image", repeats=1.5)
    with pytest.raises(ValueError, match="repeats"):
        effect_stability(adapter, {}, sample(), layer=0, group="image", repeats=0)


def test_build_causal_effect_result_rejects_duplicate_groups():
    adapter = MockAdapter()
    inputs = adapter.prepare_inputs("clean.jpg", "Question?")

    with pytest.raises(ValueError, match="duplicates"):
        build_causal_effect_result(adapter, inputs, sample(), groups=("image", "image"))


def test_build_and_write_causal_effect_validates_schema(tmp_path):
    adapter = MockAdapter(clean_logit=3.0, layer_count=4)
    inputs = adapter.prepare_inputs("clean.jpg", "Question?")
    result = build_causal_effect_result(adapter, inputs, sample())
    out_path = tmp_path / "nested" / "causal_effect_sample_001.json"

    write_causal_effect(result, out_path)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))

    assert loaded == result
    assert loaded["C_by_layer"]["image"] == pytest.approx({
        "0": 0.2,
        "1": 0.4,
        "2": 0.6,
        "3": 0.8,
    })
    assert loaded["C_by_layer"]["text"] == pytest.approx({
        "0": 1.0,
        "1": 1.2,
        "2": 1.4,
        "3": 1.6,
    })
    assert loaded["stability"]["image"]["0"]["std"] == pytest.approx(0.0)
    assert loaded["settings"] == {
        "mode": "ablate",
        "layer_count": 4,
        "groups": ["image", "text"],
        "stability_layer": 0,
        "stability_repeats": 3,
    }
    Draft202012Validator(load_causal_effect_schema()).validate(loaded)


def test_causal_effect_for_manifest_writes_one_file_per_sample(tmp_path):
    manifest_path = tmp_path / "samples.jsonl"
    records = [
        sample(sample_id="sample_a", image_path="clean_a.jpg"),
        sample(sample_id="sample_b", image_path="clean_b.jpg"),
    ]
    manifest_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    adapter = MockAdapter(clean_logit=3.0, layer_count=4)

    paths = causal_effect_for_manifest(adapter, manifest_path, tmp_path / "out")

    assert [path.split("/")[-1] for path in paths] == [
        "causal_effect_sample_a.json",
        "causal_effect_sample_b.json",
    ]
    assert adapter.prepared_images == ["clean_a.jpg", "clean_b.jpg"]
    assert [json.loads(open(path, encoding="utf-8").read())["sample_id"] for path in paths] == [
        "sample_a",
        "sample_b",
    ]
