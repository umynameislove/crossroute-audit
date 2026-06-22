from __future__ import annotations

import json
import os

import pytest

from crossroute_audit.io.analysis import (
    cliffs_between_models,
    comparison_table,
    load_route,
    per_sample_metrics,
    summarize,
)


def _write(model_dir, sid, attr, causal):
    os.makedirs(os.path.join(model_dir, "attr"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "causal"), exist_ok=True)
    with open(
        os.path.join(model_dir, "attr", f"attribution_mass_{sid}.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump({"attribution_mass": {"image": attr, "text": {}}}, handle)
    with open(
        os.path.join(model_dir, "causal", f"causal_effect_{sid}.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump({"C_by_layer": {"image": causal, "text": {}}}, handle)


def _mk(vals):
    return {str(i): value for i, value in enumerate(vals)}


def test_load_and_per_sample_metrics(tmp_path):
    model_dir = str(tmp_path / "llava")
    _write(model_dir, "s1", _mk([8, 6, 2, 1]), _mk([9, 5, 3, 1]))
    _write(model_dir, "s2", _mk([7, 5, 2, 1]), _mk([6, 6, 2, 1]))
    _write(model_dir, "s3", _mk([1, 2, 6, 8]), _mk([2, 1, 5, 9]))

    attr, causal = load_route(model_dir)
    assert set(attr) == {"s1", "s2", "s3"}

    metrics = per_sample_metrics(attr, causal, k=2)
    assert metrics["s1"]["scalar"] == pytest.approx(1.0)
    assert metrics["s3"]["scalar"] == pytest.approx(0.8)
    assert metrics["s1"]["detrended"] == pytest.approx(1.0)
    assert metrics["s2"]["detrended"] == pytest.approx(0.8)
    assert all(metrics[sample_id]["topk"] == 1.0 for sample_id in metrics)


def test_summarize_and_table(tmp_path):
    model_dir = str(tmp_path / "m")
    for index, sample_id in enumerate(["a", "b", "c"]):
        _write(
            model_dir,
            sample_id,
            _mk([8 - index, 6, 2, 1]),
            _mk([9 - index, 5, 3, 1]),
        )

    attr, causal = load_route(model_dir)
    metrics = per_sample_metrics(attr, causal)
    summary = summarize(metrics, "scalar")

    assert summary["n"] == 3
    assert 0.0 <= summary["frac_negative"] <= 1.0
    assert summary["ci"][0] <= summary["median"] <= summary["ci"][1]
    table = comparison_table({"m": summary})
    assert "median scalar" in table
    assert "| m |" in table


def test_cliffs_between_models_separates(tmp_path):
    model_a = str(tmp_path / "A")
    model_b = str(tmp_path / "B")
    _write(model_a, "x", _mk([8, 6, 2, 1]), _mk([9, 5, 3, 1]))
    _write(model_b, "y", _mk([1, 2, 6, 8]), _mk([9, 5, 3, 1]))

    per_a = per_sample_metrics(*load_route(model_a))
    per_b = per_sample_metrics(*load_route(model_b))
    assert cliffs_between_models(per_a, per_b, "scalar") == 1.0


def test_load_route_skips_missing_counterpart_and_route(tmp_path):
    model_dir = str(tmp_path / "m")
    _write(model_dir, "complete", _mk([1, 2]), _mk([1, 2]))

    attr_dir = tmp_path / "m" / "attr"
    with (attr_dir / "attribution_mass_missing.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump({"attribution_mass": {"image": _mk([1, 2])}}, handle)

    _write(model_dir, "empty_route", {}, _mk([1, 2]))

    attr, causal = load_route(model_dir)
    assert set(attr) == set(causal) == {"complete"}


def test_per_sample_metrics_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="same sample ids"):
        per_sample_metrics({}, {})
    with pytest.raises(ValueError, match="positive integer"):
        per_sample_metrics({"s": _mk([1, 2])}, {"s": _mk([1, 2])}, k=0)


def test_summarize_rejects_all_none():
    with pytest.raises(ValueError, match="no finite values"):
        summarize({"s": {"scalar": None}}, "scalar")
