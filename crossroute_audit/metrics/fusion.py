"""Fusion-based faithfulness scoring."""
from __future__ import annotations

import math


def faithfulness_score(
    rank_alignment_image: float | None,
    completeness_residual_mean: float,
    control_clean: bool,
    *,
    eps: float = 1e-9,
) -> dict:
    """Combine alignment, completeness, and control evidence into one score."""
    if rank_alignment_image is not None and (
        not math.isfinite(rank_alignment_image)
        or not -1 <= rank_alignment_image <= 1
    ):
        raise ValueError("rank_alignment_image must be None or within [-1, 1]")
    if not math.isfinite(completeness_residual_mean):
        raise ValueError("completeness_residual_mean must be finite")
    if not isinstance(control_clean, bool):
        raise ValueError("control_clean must be a boolean")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite number")

    align_term = (
        0.5
        if rank_alignment_image is None
        else (rank_alignment_image + 1) / 2
    )
    complete_term = 1 / (1 + max(completeness_residual_mean, 0))
    base = math.sqrt(align_term * complete_term)
    control_multiplier = 1.0 if control_clean else 0.5
    score = min(1.0, max(0.0, base * control_multiplier))
    flag = "ok" if control_clean else "control_unverified"

    return {
        "score": score,
        "components": {
            "align_term": align_term,
            "complete_term": complete_term,
            "base": base,
            "control_multiplier": control_multiplier,
        },
        "flag": flag,
    }
