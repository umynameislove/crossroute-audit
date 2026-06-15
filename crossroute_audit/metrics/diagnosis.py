"""Control-gated diagnosis policy."""
from __future__ import annotations

import math


def diagnose(
    control_status: dict,
    causal_effect: dict,
    attribution: dict,
    rank_alignment: dict,
    residual: float | None = None,
    attr_thresh: float = 1.0,
    causal_thresh: float = 0.5,
    align_thresh: float = 0.0,
    routing_proxy: dict | None = None,
    route_drop_thresh: float = 0.5,
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

    route_break_layer = _route_break_layer(
        routing_proxy,
        img_causal,
        route_drop_thresh,
    )
    if route_break_layer is not None:
        return {
            "diagnosis": "route_break",
            "confidence": "low" if low_conf else "high",
            "reasons": [
                f"routing proxy and image causal effect drop sharply near layer {route_break_layer}"
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


def _route_break_layer(
    routing_proxy: dict | None,
    image_causal: dict,
    drop_thresh: float,
) -> int | None:
    if not routing_proxy or not image_causal:
        return None

    if not math.isfinite(float(drop_thresh)) or drop_thresh <= 0:
        raise ValueError("route_drop_thresh must be a positive finite number")

    proxy_series = routing_proxy.get("image", routing_proxy)
    proxy_drops = _sharp_drop_layers(proxy_series, drop_thresh)
    causal_drops = _sharp_drop_layers(image_causal, drop_thresh)
    common_layers = sorted(proxy_drops & causal_drops)
    return common_layers[0] if common_layers else None


def _sharp_drop_layers(series: dict, drop_thresh: float) -> set[int]:
    items = sorted(
        ((int(layer), float(value)) for layer, value in series.items()),
        key=lambda item: item[0],
    )
    drops = set()
    for (_, previous), (layer, current) in zip(items, items[1:]):
        if not math.isfinite(previous) or not math.isfinite(current):
            raise ValueError("route-break inputs must be finite")
        baseline = max(abs(previous), 1e-12)
        if (previous - current) / baseline >= drop_thresh:
            drops.add(layer)
    return drops
