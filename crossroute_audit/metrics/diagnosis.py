"""Control-gated diagnosis policy."""
from __future__ import annotations


def diagnose(
    control_status: dict,
    causal_effect: dict,
    attribution: dict,
    rank_alignment: dict,
    residual: float | None = None,
    attr_thresh: float = 1.0,
    causal_thresh: float = 0.5,
    align_thresh: float = 0.0,
) -> dict:
    """Return a control-gated diagnosis for one sample."""
    reasons = []
    controls = control_status.get("controls", {})
    text_only = controls.get("text_only", {}).get("answerable")
    neg = controls.get("negative_control", {}).get("effect", 0.0)

    if text_only == "yes":
        return {
            "diagnosis": "language_prior",
            "confidence": "high",
            "reasons": ["text_only answerable=yes (model answers without image)"],
        }

    if abs(neg) > 0.05:
        reasons.append("negative_control not clean")

    img_attr = attribution.get("image", {})
    img_causal = causal_effect.get("image", {})
    align_img = rank_alignment.get("image")
    high_attr = (max(img_attr.values()) if img_attr else 0) >= attr_thresh
    low_causal = (max(img_causal.values()) if img_causal else 0) < causal_thresh
    low_align = (align_img is not None) and (align_img < align_thresh)
    low_conf = (
        (residual is not None and residual > 0.2)
        or align_img is None
        or bool(reasons)
    )

    if high_attr and low_causal and low_align and not low_conf:
        return {
            "diagnosis": "false_attribution_persistence",
            "confidence": "high",
            "reasons": [
                "high image attribution + low image causal + low rank alignment + gates clean"
            ],
        }

    if img_causal and sum(1 for value in img_causal.values() if value < causal_thresh) > (
        len(img_causal) * 0.7
    ):
        return {
            "diagnosis": "modality_drop",
            "confidence": "low" if low_conf else "high",
            "reasons": ["image causal effect low across most layers"],
        }

    return {
        "diagnosis": "low_confidence" if low_conf else "no_flag",
        "confidence": "low" if low_conf else "high",
        "reasons": reasons or ["no strong mismatch"],
    }
