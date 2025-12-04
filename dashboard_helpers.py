import numpy as np

# Return the index of years closest to the chosen year.
def select_year_index(years, year_choice: int) -> int:
    years_arr = np.asarray(years)
    return int(np.abs(years_arr - year_choice).argmin())


# Summarize a fan of paths (median/p95/p05) at a specific year.
def summarize_paths_at_year(years, paths: np.ndarray, year_choice: int) -> dict:
    idx = select_year_index(years, year_choice)
    slice_vals = paths[:, idx]
    return {
        "year": int(years[idx]),
        "median": float(np.median(slice_vals)),
        "p95": float(np.percentile(slice_vals, 95)),
        "p05": float(np.percentile(slice_vals, 5)),
        "idx": idx,
    }


# Compound annual growth rate from base_value to current_value over years_elapsed.
def growth_rate_annualized(base_value: float, current_value: float, years_elapsed: int) -> float:
    years_elapsed = max(years_elapsed, 1)
    if base_value <= 0:
        return 0.0
    return (current_value / base_value) ** (1.0 / years_elapsed) - 1.0


# Return 5/50/95 percentiles over time for a fan of paths.
def quantiles_over_time(paths: np.ndarray) -> dict:
    return {
        "p05": np.percentile(paths, 5, axis=0),
        "p50": np.median(paths, axis=0),
        "p95": np.percentile(paths, 95, axis=0),
    }


# Percent change helper (None if reference is zero).
def pct_delta(current: float, reference: float) -> float | None:
    if reference == 0:
        return None
    return (current - reference) / reference


# Format a value in millions of USD with compact units (M/B).
def format_musd(value: float, decimals: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "–"
    sign = "-" if value < 0 else ""
    abs_val = abs(float(value))
    if abs_val >= 1000:
        return f"{sign}${abs_val/1000:.{decimals}f}B"
    if abs_val >= 1:
        return f"{sign}${abs_val:.0f}M"
    return f"{sign}${abs_val:.3f}M"


# Human-friendly percent string with optional sign.
def format_pct(value: float | None, decimals: int = 1, show_sign: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "–"
    pct_val = value * 100
    sign = ""
    if show_sign:
        sign = "+" if pct_val >= 0 else "-"
        pct_val = abs(pct_val)
    return f"{sign}{pct_val:.{decimals}f}%"
