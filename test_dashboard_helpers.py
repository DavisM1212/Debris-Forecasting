import numpy as np

from dashboard_helpers import (
    format_musd,
    format_pct,
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
    assert summary["median"] == 12.0  # Median of the 2021 column should be 12.0
    assert summary["p95"] > summary["median"]
    assert summary["p05"] < summary["median"]


def test_growth_rate_annualized_handles_zero_base():
    assert growth_rate_annualized(0, 100, 5) == 0.0
    rate = growth_rate_annualized(100, 200, 5)
    assert 0.14 < rate < 0.15  # CAGR should land around 14.9%


def test_quantiles_over_time_shapes():
    paths = np.array([[1, 2, 3], [2, 3, 4]])
    q = quantiles_over_time(paths)
    assert set(q.keys()) == {"p05", "p50", "p95"}
    assert q["p50"].shape == (3,)


def test_pct_delta_handles_zero_reference():
    assert pct_delta(100, 0) is None
    assert pct_delta(110, 100) == 0.10


def test_format_musd_handles_scales_and_sign():
    assert format_musd(250) == "$250M"
    assert format_musd(1500) == "$1.5B"
    assert format_musd(-75) == "-$75M"
    assert format_musd(0.25) == "$0.250M"


def test_format_pct_outputs_sign_when_requested():
    assert format_pct(0.256, decimals=1) == "25.6%"
    assert format_pct(-0.034, decimals=1, show_sign=True) == "-3.4%"
    assert format_pct(0.12, decimals=0, show_sign=True) == "+12%"
