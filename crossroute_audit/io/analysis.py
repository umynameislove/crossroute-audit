"""Aggregate per-sample faithfulness metrics across models from saved artifacts."""
from __future__ import annotations

import glob
import json
import os
import statistics

from scipy.stats import wilcoxon

from crossroute_audit.metrics.rank_alignment import rank_alignment_by_group
from crossroute_audit.metrics.stats import bootstrap_ci_median, cliffs_delta
from crossroute_audit.metrics.structure_align import (
    detrended_rank_alignment,
    topk_overlap,
)


def load_route(model_dir: str, route: str = "image") -> tuple[dict, dict]:
    """Load attribution and causal layer maps for samples present in both.

    Reads ``<model_dir>/attr/attribution_mass_*.json`` and matching
    ``<model_dir>/causal/causal_effect_*.json`` files. Samples with a missing
    counterpart or a missing/empty requested route are skipped.
    """
    attr: dict = {}
    causal: dict = {}
    pattern = os.path.join(model_dir, "attr", "attribution_mass_*.json")
    for attr_path in sorted(glob.glob(pattern)):
        filename = os.path.basename(attr_path)
        sample_id = filename[len("attribution_mass_") : -len(".json")]
        causal_path = os.path.join(
            model_dir,
            "causal",
            f"causal_effect_{sample_id}.json",
        )
        if not os.path.isfile(causal_path):
            continue

        with open(attr_path, encoding="utf-8") as handle:
            attr_artifact = json.load(handle)
        with open(causal_path, encoding="utf-8") as handle:
            causal_artifact = json.load(handle)

        attr_route = attr_artifact["attribution_mass"].get(route)
        causal_route = causal_artifact["C_by_layer"].get(route)
        if attr_route and causal_route:
            attr[sample_id] = attr_route
            causal[sample_id] = causal_route
    return attr, causal


def per_sample_metrics(attr: dict, causal: dict, *, k: int = 5) -> dict:
    """Return scalar, detrended, and top-k metrics for one route."""
    if not attr or set(attr) != set(causal):
        raise ValueError("attr and causal must share the same sample ids")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")

    first_layers = len(next(iter(attr.values())))
    if first_layers == 0:
        raise ValueError("layer series must not be empty")
    effective_k = min(k, first_layers)

    detrended = detrended_rank_alignment(attr, causal)
    out = {}
    for sample_id in attr:
        scalar = rank_alignment_by_group(
            {"image": attr[sample_id]},
            {"image": causal[sample_id]},
        ).get("image")
        out[sample_id] = {
            "scalar": scalar,
            "detrended": detrended[sample_id],
            "topk": topk_overlap(
                attr[sample_id],
                causal[sample_id],
                k=effective_k,
            ),
        }
    return out


def summarize(per_sample: dict, metric: str) -> dict:
    """Summarize one metric, skipping samples where it is undefined."""
    values = [
        metrics[metric]
        for metrics in per_sample.values()
        if metrics[metric] is not None
    ]
    if not values:
        raise ValueError(f"no finite values for metric {metric!r}")

    summary = {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "frac_negative": sum(value < 0 for value in values) / len(values),
        "ci": bootstrap_ci_median(values),
    }
    summary["wilcoxon_p"] = (
        float(wilcoxon(values).pvalue)
        if len(values) >= 6 and any(value != 0 for value in values)
        else None
    )
    return summary


def cliffs_between_models(per_a: dict, per_b: dict, metric: str) -> float:
    """Return Cliff's delta for one metric between two models."""
    values_a = [
        metrics[metric]
        for metrics in per_a.values()
        if metrics[metric] is not None
    ]
    values_b = [
        metrics[metric]
        for metrics in per_b.values()
        if metrics[metric] is not None
    ]
    return cliffs_delta(values_a, values_b)


def comparison_table(summaries_by_model: dict, metric: str = "scalar") -> str:
    """Return a Markdown table comparing models on one metric."""
    lines = [
        f"| model | n | median {metric} | %neg | Wilcoxon p |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, summary in summaries_by_model.items():
        pvalue = summary["wilcoxon_p"]
        rendered_pvalue = "n/a" if pvalue is None else f"{pvalue:.2e}"
        lines.append(
            f"| {model} | {summary['n']} | {summary['median']:.3f} | "
            f"{100 * summary['frac_negative']:.1f}% | {rendered_pvalue} |"
        )
    return "\n".join(lines)
