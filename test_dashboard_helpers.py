import numpy as np

from dashboard_helpers import (
    growth_rate_annualized,
    pct_delta,
    quantiles_over_time,
    select_year_index,
    summarize_paths_at_year,
)


def test_select_year_index_picks_closest():
    years = np.array([2020, 2025, 2030])
    assert select_year_index(years, 2026) == 1
    assert select_year_index(years, 2019) == 0
    assert select_year_index(years, 2035) == 2


def test_summarize_paths_at_year_returns_quantiles():
    years = np.array([2020, 2021, 2022])
    paths = np.array(
        [
            [10, 12, 14],
            [11, 13, 17],
            [9, 11, 13],
        ]
    )
    summary = summarize_paths_at_year(years, paths, 2021)
    assert summary["year"] == 2021
    assert summary["median"] == 12.0  # median of column 2021
    assert summary["p95"] > summary["median"]
    assert summary["p05"] < summary["median"]


def test_growth_rate_annualized_handles_zero_base():
    assert growth_rate_annualized(0, 100, 5) == 0.0
    rate = growth_rate_annualized(100, 200, 5)
    assert 0.14 < rate < 0.15  # around 14.9%


def test_quantiles_over_time_shapes():
    paths = np.array([[1, 2, 3], [2, 3, 4]])
    q = quantiles_over_time(paths)
    assert set(q.keys()) == {"p05", "p50", "p95"}
    assert q["p50"].shape == (3,)


def test_pct_delta_handles_zero_reference():
    assert pct_delta(100, 0) is None
    assert pct_delta(110, 100) == 0.10
