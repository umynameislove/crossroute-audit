"""Write per-sample audit reports, figures, and the short report."""
from __future__ import annotations

import statistics
from pathlib import Path


def _norm(series: dict):
    """Return z-scored values sorted by numeric layer key."""
    layers = sorted(series, key=lambda layer: int(layer))
    values = [series[layer] for layer in layers]
    if not values:
        return layers, []
    mean = statistics.mean(values)
    std = statistics.pstdev(values) or 1.0
    return layers, [(value - mean) / std for value in values]


def plot_image_route_alignment(
    attribution_image: dict,
    causal_image: dict,
    rho,
    out_path,
    title="",
):
    """Plot normalized image-route attribution and causal trends by layer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attribution_layers, attribution_values = _norm(attribution_image)
    causal_layers, causal_values = _norm(causal_image)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [int(layer) for layer in attribution_layers],
        attribution_values,
        marker="o",
        label="AttributionMass (image)",
    )
    ax.plot(
        [int(layer) for layer in causal_layers],
        causal_values,
        marker="s",
        label="CausalEffect (image)",
    )
    ax.set_xlabel("Audit layer")
    ax.set_ylabel("z-score")
    ax.legend()
    ax.set_title(f"{title}  RankAlignment(image)={rho}")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=350)
    plt.close(fig)


def _fmt(value):
    """Render a Spearman value: 3-decimal float, or the raw value when undefined."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return value


def results_table(audit_reports: list[dict]) -> str:
    """Return a markdown summary table from audit-report payloads."""
    lines = [
        "| sample | diagnosis | rank_align(image) | rank_align(text) |",
        "|---|---|---|---|",
    ]
    for report in audit_reports:
        rank_alignment = report.get("rank_alignment", {})
        lines.append(
            f"| {report['sample_id']} | {report['diagnosis']['diagnosis']} | "
            f"{_fmt(rank_alignment.get('image'))} | {_fmt(rank_alignment.get('text'))} |"
        )
    return "\n".join(lines)
