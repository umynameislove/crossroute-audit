from __future__ import annotations

from crossroute_audit.io.report import plot_image_route_alignment, results_table


def test_results_table_returns_markdown_summary():
    reports = [
        {
            "sample_id": "sample_001",
            "diagnosis": {"diagnosis": "false_attribution_persistence"},
            "rank_alignment": {"image": -1.0, "text": 1.0},
        },
        {
            "sample_id": "sample_002",
            "diagnosis": {"diagnosis": "language_prior"},
            "rank_alignment": {"image": None},
        },
    ]

    assert results_table(reports) == "\n".join(
        [
            "| sample | diagnosis | rank_align(image) | rank_align(text) |",
            "|---|---|---|---|",
            "| sample_001 | false_attribution_persistence | -1.000 | 1.000 |",
            "| sample_002 | language_prior | None | None |",
        ]
    )


def test_plot_image_route_alignment_writes_png(tmp_path):
    out_path = tmp_path / "figures" / "image_route_alignment.png"

    plot_image_route_alignment(
        attribution_image={"0": 1.0, "1": 2.0, "10": 3.0},
        causal_image={"0": 3.0, "1": 2.0, "10": 1.0},
        rho=-1.0,
        out_path=out_path,
        title="sample_001",
    )

    assert out_path.is_file()
    assert out_path.stat().st_size > 0
