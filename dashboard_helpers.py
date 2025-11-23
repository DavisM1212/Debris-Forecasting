import numpy as np


def select_year_index(years, year_choice: int) -> int:
    """Return the index of years closest to the chosen year."""
    years_arr = np.asarray(years)
    return int(np.abs(years_arr - year_choice).argmin())


def summarize_paths_at_year(years, paths: np.ndarray, year_choice: int) -> dict:
    """Summarize a fan of paths (median/p95/p05) at a specific year."""
    idx = select_year_index(years, year_choice)
    slice_vals = paths[:, idx]
    return {
        "year": int(years[idx]),
        "median": float(np.median(slice_vals)),
        "p95": float(np.percentile(slice_vals, 95)),
        "p05": float(np.percentile(slice_vals, 5)),
        "idx": idx,
    }


def growth_rate_annualized(base_value: float, current_value: float, years_elapsed: int) -> float:
    """Compound annual growth rate from base_value to current_value over years_elapsed."""
    years_elapsed = max(years_elapsed, 1)
    if base_value <= 0:
        return 0.0
    return (current_value / base_value) ** (1.0 / years_elapsed) - 1.0


def quantiles_over_time(paths: np.ndarray) -> dict:
    """Return 5/50/95 percentiles over time for a fan of paths."""
    return {
        "p05": np.percentile(paths, 5, axis=0),
        "p50": np.median(paths, axis=0),
        "p95": np.percentile(paths, 95, axis=0),
    }


def pct_delta(current: float, reference: float) -> float | None:
    """Percent change helper (None if reference is zero)."""
    if reference == 0:
        return None
    return (current - reference) / reference
