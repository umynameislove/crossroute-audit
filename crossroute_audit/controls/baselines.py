"""Baseline control runs that gate false-attribution diagnoses.

These controls separate genuine false attribution from language-prior and
redundancy confounds. They depend only on the ``ModelAdapter`` interface, so they
can be developed and tested against a mock adapter without loading a real model.

``control_status`` (returned by :func:`run_controls`) shape::

    {
      "text_only":        {"target_logit": float, "answerable": "yes"|"no"},
      "no_image":         {"target_logit": float, "answerable": "yes"|"no"},
      "counterfactual":   {"delta_logit_cf": float|None, "flipped": bool|None,
                           "expected_flip": bool|None},
      "negative_control": {"effect": float},
    }

Thresholds are heuristics for the MVP and should be tuned on real data.
"""
from __future__ import annotations

from PIL import Image

# Target logit still >= this fraction of the clean logit => answerable without image.
ANSWERABLE_KEEP_RATIO = 0.5
# clean - counterfactual logit >= this => target considered "flipped".
FLIP_THRESHOLD = 1.0

_BLANK_IMAGE = Image.new("RGB", (224, 224), (0, 0, 0))


def _target_logit(adapter, image, sample) -> float:
    inputs = adapter.prepare_inputs(image, sample["question"])
    return adapter.get_target_logit(inputs, sample["target_answer"], sample["target_token_policy"])


def _clean_logit(adapter, sample) -> float:
    return _target_logit(adapter, sample["image_path"], sample)


def _answerable(logit: float, clean: float) -> str:
    return "yes" if logit >= ANSWERABLE_KEEP_RATIO * clean else "no"


def run_text_only(adapter, sample) -> dict:
    """Drop the image (pass ``None``), keep the question; high retained logit => language prior."""
    clean = _clean_logit(adapter, sample)
    logit = _target_logit(adapter, None, sample)
    return {"target_logit": logit, "answerable": _answerable(logit, clean)}


def run_no_image(adapter, sample) -> dict:
    """Replace the image with a blank image; high retained logit => language prior."""
    clean = _clean_logit(adapter, sample)
    logit = _target_logit(adapter, _BLANK_IMAGE, sample)
    return {"target_logit": logit, "answerable": _answerable(logit, clean)}


def run_counterfactual(adapter, sample) -> dict:
    """Swap in the counterfactual image that should flip the target."""
    clean = _clean_logit(adapter, sample)
    cf_path = sample.get("counterfactual_image_path")
    expected = sample.get("expected_flip")
    if not cf_path:
        return {"delta_logit_cf": None, "flipped": None, "expected_flip": expected}
    cf_logit = _target_logit(adapter, cf_path, sample)
    delta = clean - cf_logit
    return {"delta_logit_cf": delta, "flipped": bool(delta >= FLIP_THRESHOLD), "expected_flip": expected}


def run_negative_control(adapter, sample) -> dict:
    """Intervene on an irrelevant region; the effect should be near zero."""
    inputs = adapter.prepare_inputs(sample["image_path"], sample["question"])
    clean = adapter.get_target_logit(inputs, sample["target_answer"], sample["target_token_policy"])
    intervened = adapter.intervene(inputs, layer=0, group="image", mode="negative_control")
    return {"effect": clean - intervened}


_CONTROLS = {
    "text_only": run_text_only,
    "no_image": run_no_image,
    "counterfactual": run_counterfactual,
    "negative_control": run_negative_control,
}


def run_controls(adapter, sample, which=None) -> dict:
    """Run the requested controls and return the ``control_status`` dict.

    ``which`` is ``None`` (all controls) or a comma-separated string / iterable of names.
    """
    if which is None:
        names = list(_CONTROLS)
    elif isinstance(which, str):
        names = [w.strip() for w in which.split(",") if w.strip()]
    else:
        names = list(which)
    return {name: _CONTROLS[name](adapter, sample) for name in names if name in _CONTROLS}
