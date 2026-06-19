from __future__ import annotations

import pytest

from crossroute_audit.io.report import comparison_table, multi_model_summary


def _report(rho, diagnosis):
    return {
        "rank_alignment": {"image": rho},
        "diagnosis": {"diagnosis": diagnosis},
    }


def test_multi_model_summary_calculates_known_values():
    summary = multi_model_summary(
        {
            "blip2": [
                _report(-0.8, "language_prior"),
                _report(-0.2, "language_prior"),
                _report(0.4, "no_flag"),
                _report(None, "no_flag"),
            ],
            "llava": [
                _report(0.1, "no_flag"),
                _report(0.5, "no_flag"),
            ],
        }
    )

    assert summary["blip2"] == {
        "n": 4,
        "median_rho": pytest.approx(-0.2),
        "frac_negative": pytest.approx(2 / 3),
        "diagnosis_counts": {"language_prior": 2, "no_flag": 2},
    }
    assert summary["llava"] == {
        "n": 2,
        "median_rho": pytest.approx(0.3),
        "frac_negative": 0.0,
        "diagnosis_counts": {"no_flag": 2},
    }


def test_multi_model_summary_counts_n_but_excludes_none_rho():
    summary = multi_model_summary(
        {"model": [_report(None, "no_flag"), _report(None, "route_break")]}
    )

    assert summary["model"]["n"] == 2
    assert summary["model"]["median_rho"] is None
    assert summary["model"]["frac_negative"] is None


def test_comparison_table_contains_models_and_rounded_values():
    table = comparison_table(
        {
            "blip2": {
                "n": 4,
                "median_rho": -0.23456,
                "frac_negative": 2 / 3,
                "diagnosis_counts": {"no_flag": 1, "language_prior": 3},
            },
            "llava": {
                "n": 2,
                "median_rho": 0.34567,
                "frac_negative": 0.0,
                "diagnosis_counts": {"no_flag": 2},
            },
        }
    )

    assert "| blip2 | 4 | -0.235 | 66.667% | language_prior |" in table
    assert "| llava | 2 | 0.346 | 0.000% | no_flag |" in table


def test_comparison_table_handles_empty_model_summary():
    summary = multi_model_summary({"empty": []})

    assert (
        "| empty | 0 | None | None | None |"
        in comparison_table(summary)
    )
