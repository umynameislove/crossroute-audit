from __future__ import annotations

import json

from jsonschema import Draft202012Validator
import pytest

from crossroute_audit.io.control_status import (
    build_control_status,
    control_status_for_manifest,
    write_control_status,
)


class MockAdapter:
    def __init__(self, logits: dict[str, float], negative_control_effect: float = 0.0):
        self.logits = logits
        self.negative_control_effect = negative_control_effect
        self.interventions = []

    def prepare_inputs(self, image, question: str):
        return {"image": image, "question": question}

    def get_target_logit(self, inputs, target_answer: str, policy: str) -> float:
        return self._logit_for_image(inputs["image"])

    def intervene(self, inputs, layer: int, group: str, mode: str):
        self.interventions.append((layer, group, mode))
        return self._logit_for_image(inputs["image"]) - self.negative_control_effect

    def _logit_for_image(self, image) -> float:
        if image is None:
            key = "text_only"
        elif hasattr(image, "getpixel"):
            key = "blank"
        else:
            key = str(image)
        return self.logits[key]


def sample(**overrides) -> dict:
    base = {
        "sample_id": "sample_001",
        "image_path": "clean.jpg",
        "question": "Is the target present?",
        "target_answer": "yes",
        "target_token_policy": "first_generated_token",
        "expected_visual_dependency": "high",
        "text_only_answerable": "no",
        "counterfactual_image_path": "counterfactual.jpg",
        "control_type": "clean",
        "expected_flip": True,
        "label": None,
        "notes": "Mock sample for control-status writer tests.",
    }
    base.update(overrides)
    return base


def load_control_status_schema() -> dict:
    with open("schemas/control_status.schema.json", encoding="utf-8") as schema_file:
        return json.load(schema_file)


def test_build_control_status_wraps_all_controls_and_metadata():
    adapter = MockAdapter(
        {
            "clean.jpg": 4.0,
            "text_only": 0.3,
            "blank": 0.2,
            "counterfactual.jpg": 2.5,
        },
        negative_control_effect=0.05,
    )

    result = build_control_status(adapter, sample())

    assert result["sample_id"] == "sample_001"
    assert result["target_answer"] == "yes"
    assert result["which"] == [
        "text_only",
        "no_image",
        "counterfactual",
        "negative_control",
    ]
    assert set(result["controls"]) == {
        "text_only",
        "no_image",
        "counterfactual",
        "negative_control",
    }
    assert result["controls"]["text_only"] == {"target_logit": 0.3, "answerable": "no"}
    assert result["controls"]["no_image"] == {"target_logit": 0.2, "answerable": "no"}
    assert result["controls"]["counterfactual"]["delta_logit_cf"] == pytest.approx(1.5)
    assert result["controls"]["counterfactual"]["flipped"] is True
    assert result["controls"]["negative_control"]["effect"] == pytest.approx(0.05)
    assert adapter.interventions == [(0, "image", "negative_control")]


def test_write_control_status_round_trips_json_and_validates_schema(tmp_path):
    status = build_control_status(
        MockAdapter(
            {
                "clean.jpg": 3.0,
                "text_only": 0.1,
                "blank": 0.1,
                "counterfactual.jpg": 1.2,
            },
            negative_control_effect=0.0,
        ),
        sample(),
    )
    out_path = tmp_path / "nested" / "control_status_sample_001.json"

    write_control_status(status, out_path)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))

    assert loaded == status
    Draft202012Validator(load_control_status_schema()).validate(loaded)


def test_build_control_status_rejects_unknown_control_name():
    adapter = MockAdapter({"clean.jpg": 3.0})

    with pytest.raises(ValueError, match="unsupported control"):
        build_control_status(adapter, sample(), which=("text_only", "typo_control"))


def test_control_status_for_manifest_writes_one_file_per_sample(tmp_path):
    manifest_path = tmp_path / "samples.jsonl"
    records = [
        sample(sample_id="sample_a", image_path="clean_a.jpg", counterfactual_image_path=None, expected_flip=None),
        sample(sample_id="sample_b", image_path="clean_b.jpg", counterfactual_image_path="counterfactual_b.jpg"),
    ]
    manifest_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    adapter = MockAdapter(
        {
            "clean_a.jpg": 4.0,
            "clean_b.jpg": 5.0,
            "text_only": 0.1,
            "blank": 0.1,
            "counterfactual_b.jpg": 2.0,
        },
        negative_control_effect=0.0,
    )

    paths = control_status_for_manifest(adapter, manifest_path, tmp_path / "out")

    assert [path.split("/")[-1] for path in paths] == [
        "control_status_sample_a.json",
        "control_status_sample_b.json",
    ]
    assert [json.loads(open(path, encoding="utf-8").read())["sample_id"] for path in paths] == [
        "sample_a",
        "sample_b",
    ]
