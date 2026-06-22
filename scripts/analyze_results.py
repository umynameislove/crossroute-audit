"""Create final CrossRoute-Audit comparison tables and paper figures."""
from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossroute_audit.io.analysis import (
    cliffs_between_models,
    comparison_table,
    load_route,
    per_sample_metrics,
    summarize,
)


MODEL_COLORS = (
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
)
LINE_COLORS = {
    "attribution": "#0072B2",
    "causal": "#D55E00",
}


def configure_paper_style() -> None:
    """Apply a compact, colorblind-safe paper plotting style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 350,
        }
    )


def plot_metric_boxplot(
    per_sample_by_model: dict[str, dict],
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Plot scalar and detrended rank-alignment distributions by model."""
    if not per_sample_by_model:
        raise ValueError("per_sample_by_model must not be empty")
    configure_paper_style()

    models = list(per_sample_by_model)
    metrics = ("scalar", "detrended")
    positions: list[float] = []
    data: list[list[float]] = []
    colors: list[str] = []
    hatches: list[str] = []
    tick_positions: list[float] = []

    for model_index, model in enumerate(models):
        group_start = model_index * 3.0
        group_positions = []
        for metric_index, metric in enumerate(metrics):
            values = [
                sample_metrics[metric]
                for sample_metrics in per_sample_by_model[model].values()
                if sample_metrics[metric] is not None
            ]
            if not values:
                raise ValueError(
                    f"model {model!r} has no finite values for metric {metric!r}"
                )
            position = group_start + metric_index
            positions.append(position)
            group_positions.append(position)
            data.append(values)
            colors.append(MODEL_COLORS[model_index % len(MODEL_COLORS)])
            hatches.append("" if metric == "scalar" else "//")
        tick_positions.append(statistics.mean(group_positions))

    fig, axis = plt.subplots(figsize=(max(6.0, 2.4 * len(models)), 4.2))
    boxplot = axis.boxplot(
        data,
        positions=positions,
        widths=0.72,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, color, hatch in zip(boxplot["boxes"], colors, hatches):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_hatch(hatch)

    axis.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    axis.set_xticks(tick_positions, models)
    axis.set_ylabel("Rank alignment (Spearman ρ)")
    axis.set_title("Attribution–causal alignment by model")

    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#777777",
            alpha=0.78,
            label="Scalar",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#777777",
            alpha=0.78,
            hatch="//",
            label="Detrended",
        ),
    ]
    axis.legend(handles=legend_handles, loc="best")
    fig.tight_layout()
    return save_figure(fig, Path(out_dir) / "rank_alignment_boxplot")


def plot_representative_layers(
    attr_by_model: dict[str, dict],
    causal_by_model: dict[str, dict],
    per_sample_by_model: dict[str, dict],
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Plot z-scored attribution and causal layer profiles for each model."""
    models = list(attr_by_model)
    if (
        not models
        or set(models) != set(causal_by_model)
        or set(models) != set(per_sample_by_model)
    ):
        raise ValueError("attr, causal, and metrics must share model names")
    configure_paper_style()

    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(7.2, max(3.2, 2.8 * len(models))),
        squeeze=False,
    )
    for row, model in enumerate(models):
        sample_id = select_representative_sample(per_sample_by_model[model])
        attr = attr_by_model[model][sample_id]
        causal = causal_by_model[model][sample_id]
        layers = _sorted_common_layers(attr, causal)
        x_values = [int(layer) if str(layer).lstrip("-").isdigit() else index
                    for index, layer in enumerate(layers)]
        attr_z = zscore([float(attr[layer]) for layer in layers])
        causal_z = zscore([float(causal[layer]) for layer in layers])

        axis = axes[row][0]
        axis.plot(
            x_values,
            attr_z,
            color=LINE_COLORS["attribution"],
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            label="Attribution",
        )
        axis.plot(
            x_values,
            causal_z,
            color=LINE_COLORS["causal"],
            marker="s",
            markersize=3.2,
            linewidth=1.5,
            label="Causal effect",
        )
        axis.axhline(0.0, color="#777777", linewidth=0.7, linestyle="--")
        axis.set_title(f"{model}: representative sample {sample_id}")
        axis.set_ylabel("z-score")
        axis.legend(loc="best")

    axes[-1][0].set_xlabel("Layer")
    fig.tight_layout()
    return save_figure(fig, Path(out_dir) / "representative_layer_profiles")


def select_representative_sample(per_sample: dict) -> str:
    """Select the sample whose scalar alignment is closest to the median."""
    finite = [
        (sample_id, metrics["scalar"])
        for sample_id, metrics in per_sample.items()
        if metrics["scalar"] is not None
    ]
    if not finite:
        raise ValueError("no finite scalar metric for representative sample")
    median = statistics.median(value for _, value in finite)
    return min(finite, key=lambda item: (abs(item[1] - median), item[0]))[0]


def zscore(values: list[float]) -> list[float]:
    """Return population z-scores, or zeros for a constant series."""
    if not values:
        raise ValueError("values must not be empty")
    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    if std == 0:
        return [0.0] * len(values)
    return [(value - mean) / std for value in values]


def save_figure(figure, base_path: Path) -> tuple[Path, Path]:
    """Save one figure as 350-dpi PNG and vector PDF."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=350, bbox_inches="tight")
    figure.savefig(pdf_path, dpi=350, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def analyze_models(
    model_dirs: dict[str, str],
    out_dir: str | Path,
    *,
    route: str = "image",
    k: int = 5,
) -> dict:
    """Load model artifacts, print comparisons, and create paper figures."""
    if len(model_dirs) < 2:
        raise ValueError("at least two models are required")

    attr_by_model: dict[str, dict] = {}
    causal_by_model: dict[str, dict] = {}
    metrics_by_model: dict[str, dict] = {}
    for model, model_dir in model_dirs.items():
        attr, causal = load_route(model_dir, route=route)
        if not attr:
            raise ValueError(
                f"no matched attribution/causal artifacts for model {model!r}"
            )
        attr_by_model[model] = attr
        causal_by_model[model] = causal
        metrics_by_model[model] = per_sample_metrics(attr, causal, k=k)

    for metric in ("scalar", "detrended", "topk"):
        summaries = {
            model: summarize(per_sample, metric)
            for model, per_sample in metrics_by_model.items()
        }
        print(f"\n{metric}")
        print(comparison_table(summaries, metric))
        for left, right in itertools.combinations(model_dirs, 2):
            delta = cliffs_between_models(
                metrics_by_model[left],
                metrics_by_model[right],
                metric,
            )
            print(f"Cliff's delta ({left} vs {right}): {delta:.3f}")

    boxplot_paths = plot_metric_boxplot(metrics_by_model, out_dir)
    profile_paths = plot_representative_layers(
        attr_by_model,
        causal_by_model,
        metrics_by_model,
        out_dir,
    )
    return {
        "metrics": metrics_by_model,
        "figures": {
            "boxplot": boxplot_paths,
            "profiles": profile_paths,
        },
    }


def parse_model_specs(specs: list[str]) -> dict[str, str]:
    """Parse repeated ``name=directory`` CLI model specifications."""
    parsed: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid model specification {spec!r}; use name=path")
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"invalid model specification {spec!r}; use name=path")
        if name in parsed:
            raise ValueError(f"duplicate model name: {name}")
        parsed[name] = path
    return parsed


def _sorted_common_layers(attr: dict, causal: dict) -> list[str]:
    if not attr or set(attr) != set(causal):
        raise ValueError("attribution and causal series must share non-empty layers")
    return sorted(
        attr,
        key=lambda layer: (
            0,
            int(layer),
        )
        if str(layer).lstrip("-").isdigit()
        else (1, str(layer)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="NAME=DIR",
        help="Model artifact directories, e.g. llava=runs/llava.",
    )
    parser.add_argument("--out", default="runs/figures")
    parser.add_argument("--route", default="image")
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dirs = parse_model_specs(args.models)
    analyze_models(model_dirs, args.out, route=args.route, k=args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
