import copy

import pytest

from crossroute_audit.metrics.power import (
    _wilcoxon_pvalue,
    min_n_for_power,
    power_curve,
    power_table,
    summarize_power_table,
)


STRONG_EFFECT = [0.8, 0.9, 1.0, 1.1, 1.2]
ZERO_EFFECT = [0.0, 0.0, 0.0, 0.0]
SMALL_EFFECT = [-0.01, 0.0, 0.01, 0.0, -0.01]


def test_power_curve_is_deterministic_for_seed():
    first = power_curve(STRONG_EFFECT, sizes=(20, 100), n_sim=200, seed=0)
    second = power_curve(STRONG_EFFECT, sizes=(20, 100), n_sim=200, seed=0)

    assert first == second


def test_strong_effect_has_high_power_at_large_n():
    curve = power_curve(STRONG_EFFECT, sizes=(20, 100), n_sim=300, seed=0)

    assert curve[100] >= 0.9
    assert curve[100] >= curve[20] - 0.1


def test_zero_effect_has_zero_power():
    curve = power_curve(ZERO_EFFECT, sizes=(20, 100), n_sim=200, seed=0)

    assert curve == {20: 0.0, 100: 0.0}


def test_power_curve_rejects_empty_values():
    with pytest.raises(ValueError, match="values must not be empty"):
        power_curve([])


def test_power_curve_rejects_empty_sizes():
    with pytest.raises(ValueError, match="sizes must not be empty"):
        power_curve(STRONG_EFFECT, sizes=())


def test_power_curve_rejects_invalid_size():
    with pytest.raises(ValueError, match="positive integers"):
        power_curve(STRONG_EFFECT, sizes=(0,))


@pytest.mark.parametrize("alpha", (0, 1))
def test_power_curve_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="between 0 and 1"):
        power_curve(STRONG_EFFECT, alpha=alpha)


def test_power_curve_rejects_invalid_n_sim():
    with pytest.raises(ValueError, match="positive integer"):
        power_curve(STRONG_EFFECT, n_sim=0)


def test_wilcoxon_pvalue_handles_all_zero_sample():
    assert _wilcoxon_pvalue([0, 0, 0]) == pytest.approx(1.0)


def test_min_n_for_power_returns_smallest_size_that_reaches_target():
    assert min_n_for_power(
        STRONG_EFFECT,
        target=0.8,
        sizes=(20, 100),
        n_sim=200,
        seed=0,
    ) == 20


def test_min_n_for_power_returns_none_when_target_is_not_reached():
    assert min_n_for_power(
        ZERO_EFFECT,
        target=0.8,
        sizes=(20, 100),
        n_sim=200,
        seed=0,
    ) is None


@pytest.mark.parametrize("target", (0, 1))
def test_min_n_for_power_rejects_invalid_target(target):
    with pytest.raises(ValueError, match="between 0 and 1"):
        min_n_for_power(STRONG_EFFECT, target=target)


def test_power_table_has_required_nested_keys():
    table = power_table(
        {"model_a": {"rank_alignment_gap": STRONG_EFFECT}},
        sizes=(20, 100),
        n_sim=100,
        seed=0,
    )
    entry = table["model_a"]["rank_alignment_gap"]

    assert set(entry) == {"power_curve", "min_n_for_target", "target", "alpha"}


def test_power_table_is_deterministic_for_seed():
    data = {
        "model_b": {"rank_alignment_gap": SMALL_EFFECT},
        "model_a": {"emd_gap": STRONG_EFFECT},
    }

    assert power_table(data, sizes=(20,), n_sim=100, seed=7) == power_table(
        data,
        sizes=(20,),
        n_sim=100,
        seed=7,
    )


def test_power_table_rejects_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        power_table({})


def test_power_table_rejects_empty_model_metrics():
    with pytest.raises(ValueError, match="model metrics must not be empty"):
        power_table({"model": {}})


def test_summarize_power_table_returns_report_rows():
    table = power_table(
        {"model_a": {"rank_alignment_gap": STRONG_EFFECT}},
        sizes=(20, 100),
        n_sim=100,
        seed=0,
    )
    rows = summarize_power_table(table)

    assert len(rows) == 1
    assert set(rows[0]) == {
        "model",
        "metric",
        "min_n_for_target",
        "target",
        "alpha",
        "max_power",
    }
    assert rows[0]["model"] == "model_a"
    assert rows[0]["metric"] == "rank_alignment_gap"


def test_inputs_are_not_mutated():
    values = list(STRONG_EFFECT)
    sizes = [20, 100]
    data = {
        "model_a": {
            "rank_alignment_gap": list(STRONG_EFFECT),
            "emd_gap": list(SMALL_EFFECT),
        }
    }
    data_before = copy.deepcopy(data)

    power_curve(values, sizes=sizes, n_sim=50, seed=0)
    min_n_for_power(values, sizes=sizes, n_sim=50, seed=0)
    table = power_table(data, sizes=sizes, n_sim=50, seed=0)
    summarize_power_table(table)

    assert values == STRONG_EFFECT
    assert sizes == [20, 100]
    assert data == data_before
