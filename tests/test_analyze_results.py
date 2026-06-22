from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_results import (
    analyze_models,
    parse_model_specs,
    plot_metric_boxplot,
    plot_representative_layers,
    select_representative_sample,
    zscore,
)


def _series(values):
    return {str(index): value for index, value in enumerate(values)}


def _metrics():
    return {
        "s1": {"scalar": 1.0, "detrended": 0.8, "topk": 1.0},
        "s2": {"scalar": 0.5, "detrended": -0.2, "topk": 0.5},
        "s3": {"scalar": -0.4, "detrended": 0.1, "topk": 0.5},
    }


def _write_artifact_pair(model_dir: Path, sample_id: str, attr, causal):
    attr_dir = model_dir / "attr"
    causal_dir = model_dir / "causal"
    attr_dir.mkdir(parents=True, exist_ok=True)
    causal_dir.mkdir(parents=True, exist_ok=True)
    (attr_dir / f"attribution_mass_{sample_id}.json").write_text(
        json.dumps({"attribution_mass": {"image": attr}}),
        encoding="utf-8",
    )
    (causal_dir / f"causal_effect_{sample_id}.json").write_text(
        json.dumps({"C_by_layer": {"image": causal}}),
        encoding="utf-8",
    )


def test_plot_functions_write_png_and_pdf(tmp_path):
    per_model = {"llava": _metrics(), "blip2": _metrics()}
    attr = {
        model: {
            "s1": _series([8, 6, 2, 1]),
            "s2": _series([7, 5, 3, 1]),
            "s3": _series([1, 2, 6, 8]),
        }
        for model in per_model
    }
    causal = {
        model: {
            "s1": _series([9, 5, 3, 1]),
            "s2": _series([6, 6, 2, 1]),
            "s3": _series([2, 1, 5, 9]),
        }
        for model in per_model
    }

    paths = [
        *plot_metric_boxplot(per_model, tmp_path),
        *plot_representative_layers(attr, causal, per_model, tmp_path),
    ]

    assert {path.suffix for path in paths} == {".png", ".pdf"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_analyze_models_prints_tables_and_writes_figures(tmp_path, capsys):
    model_dirs = {}
    for model, reverse in (("llava", False), ("blip2", True)):
        model_dir = tmp_path / model
        model_dirs[model] = str(model_dir)
        for index in range(6):
            attr_values = [8 + index, 6, 2, 1]
            causal_values = [9 + index, 5, 3, 1]
            if reverse:
                attr_values = list(reversed(attr_values))
            _write_artifact_pair(
                model_dir,
                f"s{index}",
                _series(attr_values),
                _series(causal_values),
            )

    result = analyze_models(model_dirs, tmp_path / "figures", k=2)
    output = capsys.readouterr().out

    assert "median scalar" in output
    assert "median detrended" in output
    assert "median topk" in output
    assert "Cliff's delta (llava vs blip2)" in output
    for figure_paths in result["figures"].values():
        assert all(path.is_file() and path.stat().st_size > 0 for path in figure_paths)


def test_parse_model_specs_and_validation():
    assert parse_model_specs(["llava=runs/llava", "blip2=runs/blip2"]) == {
        "llava": "runs/llava",
        "blip2": "runs/blip2",
    }
    with pytest.raises(ValueError, match="name=path"):
        parse_model_specs(["llava"])
    with pytest.raises(ValueError, match="duplicate model"):
        parse_model_specs(["m=a", "m=b"])


def test_zscore_and_representative_selection():
    values = zscore([1.0, 2.0, 3.0])
    assert sum(values) == pytest.approx(0.0)
    assert zscore([2.0, 2.0]) == [0.0, 0.0]
    assert select_representative_sample(_metrics()) == "s2"
