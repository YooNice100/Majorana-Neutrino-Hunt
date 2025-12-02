# src/utils/stats.py

import numpy as np
from scipy.stats import ttest_ind


# ------------------------------------------------------------
# Build strict SSE / MSE masks
# ------------------------------------------------------------
def make_strict_masks(data):
    """
    Build boolean masks for strict SSE and strict MSE events.

    data is the dictionary returned by load_hdf5() in io.py
    and must contain the four PSD label arrays.
    """
    low  = data["psd_label_low_avse"].astype(bool)
    high = data["psd_label_high_avse"].astype(bool)
    dcr  = data["psd_label_dcr"].astype(bool)
    lq   = data["psd_label_lq"].astype(bool)

    strict_sse =  low & (~high) & dcr & lq
    strict_mse = (~low) & high & (~dcr) & (~lq)

    return strict_sse, strict_mse


# ------------------------------------------------------------
# Simple helpers for stats
# ------------------------------------------------------------
def _clean(x):
    """Convert to float array and drop NaNs / infs."""
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def welch_ttest(sse, mse):
    """
    Welch two-sample t-test (SSE vs MSE).

    Returns (t_stat, p_value).
    """
    sse_clean = _clean(sse)
    mse_clean = _clean(mse)

    if len(sse_clean) == 0 or len(mse_clean) == 0:
        return np.nan, np.nan

    t_stat, p_val = ttest_ind(sse_clean, mse_clean, equal_var=False)
    return float(t_stat), float(p_val)


def summarize_feature(name, sse, mse):
    """
    Print mean/std for SSE and MSE and the Welch t-test result.
    """
    sse_clean = _clean(sse)
    mse_clean = _clean(mse)

    print()
    print(f"=== {name} ===")
    print(f"SSE: n={len(sse_clean)}, mean={np.mean(sse_clean):.4g}, std={np.std(sse_clean):.4g}")
    print(f"MSE: n={len(mse_clean)}, mean={np.mean(mse_clean):.4g}, std={np.std(mse_clean):.4g}")

    t_stat, p_val = welch_ttest(sse_clean, mse_clean)
    print(f"Welch t-test: t = {t_stat:.4g}, p = {p_val:.3e}")
