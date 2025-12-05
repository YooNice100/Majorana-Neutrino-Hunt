import numpy as np
from ..utils.transforms import estimate_baseline
from scipy.interpolate import interp1d

# ------------------------------------------------------------
# 1. Peak width between 25 percent and 75 percent of max
# ------------------------------------------------------------
def compute_peak_width_25_75(waveform):
    """
    Computes the width (in samples) between:
      - first crossing of 25 percent of peak
      - last crossing of 75 percent of peak

    Parameters
    ----------
    waveform : array-like
        Raw waveform.

    Returns
    -------
    width : float
        Number of samples between 25 percent and 75 percent crossings.
        NaN if the waveform never reaches required thresholds.
    left_idx : int or None
        First sample >= 25 percent peak.
    right_idx : int or None
        Last sample >= 75 percent peak.
    """
    y = np.asarray(waveform, dtype=float)

    peak_val = float(np.max(y))
    if peak_val <= 0:
        return np.nan, None, None

    level25 = 0.25 * peak_val
    level75 = 0.75 * peak_val

    above_25 = np.where(y >= level25)[0]
    above_75 = np.where(y >= level75)[0]

    if len(above_25) == 0 or len(above_75) == 0:
        return np.nan, None, None

    left_idx = int(above_25[0])
    right_idx = int(above_75[-1])

    width = right_idx - left_idx
    if width < 0:
        return np.nan, left_idx, right_idx

    return float(width), left_idx, right_idx



# ------------------------------------------------------------
# 2. Energy Duration (time to reach 90 percent cumulative energy)
# ------------------------------------------------------------
def compute_energy_duration(waveform, threshold=0.9):
    """
    Returns the number of samples needed to reach `threshold`
    fraction of total squared energy.

    Parameters
    ----------
    waveform : array-like
        Raw waveform.
    threshold : float
        Fraction of total energy to reach (default 0.9).

    Returns
    -------
    duration_index : int or NaN
        First index where cumulative energy reaches threshold.
    """
    y = np.asarray(waveform, dtype=float)

    # Use squared waveform as "energy"
    energy = y ** 2
    total_energy = float(np.sum(energy))
    if total_energy == 0:
        return np.nan

    cumulative = np.cumsum(energy)
    target = threshold * total_energy

    idxs = np.where(cumulative >= target)[0]
    if len(idxs) == 0:
        return np.nan

    return int(idxs[0])



# ------------------------------------------------------------
# 3. Basic tp0 Estimation (simple threshold method)
# ------------------------------------------------------------
def estimate_tp0_threshold(waveform, threshold=30):
    """
    Simple tp0 estimator:
    Returns the first index where the waveform rises above
    (baseline + threshold).

    Parameters
    ----------
    waveform : array-like
        Raw waveform.
    threshold : float
        ADC rise above baseline.

    Returns
    -------
    tp0 : int or None
        Estimated start-time of rising edge.
    """
    y = np.asarray(waveform, dtype=float)
    baseline, _ = estimate_baseline(y)

    idxs = np.where(y > (baseline + threshold))[0]
    return int(idxs[0]) if len(idxs) > 0 else None

# ------------------------------------------------------------
# 4. AvsE 
# ------------------------------------------------------------
def compute_avse(raw_waveform, energy_label, n_baseline=50):
    N = raw_waveform.shape[0]
    AvsE = np.full(N, np.nan, float)
    A    = np.full(N, np.nan, float)

    for i in range(N):
        w = raw_waveform[i].astype(float)
        E = float(energy_label[i])
        b = np.mean(w[:n_baseline])
        w0 = w - b

        cur = np.diff(w0)  # per-sample slope
        Ai = np.max(cur) if cur.size else np.nan

        A[i] = Ai
        AvsE[i] = (Ai / E) if E > 0 else np.nan

    return AvsE, A

# ------------------------------------------------------------
# 5. Drift Times
# ------------------------------------------------------------
def compute_drift_times(waveform, tp0, step=0.1):
    """
    Computes drift-time parameters by interpolating the rising edge.
    Returns NaN values if waveform is not suitable.
    """
    y = np.asarray(waveform, dtype=float)
    peak_idx = int(np.argmax(y))

    # invalid tp0
    if tp0 is None or tp0 >= peak_idx:
        return np.nan, np.nan, np.nan

    rise = y[tp0 : peak_idx + 1]
    if len(rise) < 3:
        return np.nan, np.nan, np.nan

    t = np.arange(len(rise))

    # safeguard interpolation
    try:
        interp = interp1d(t, rise, kind="linear")
        t_new = np.arange(0, len(t) - 1, step)
        if len(t_new) == 0:
            return np.nan, np.nan, np.nan

        y_new = interp(t_new)
        if len(y_new) == 0:
            return np.nan, np.nan, np.nan
    except Exception:
        return np.nan, np.nan, np.nan

    peak_val = np.max(y_new)

    # thresholds
    try:
        tdrift_999 = np.where(y_new >= 0.999 * peak_val)[0][0]
        tdrift_50  = np.where(y_new >= 0.5   * peak_val)[0][0]
        tdrift_10  = np.where(y_new >= 0.1   * peak_val)[0][0]
    except Exception:
        return np.nan, np.nan, np.nan

    return tdrift_999, tdrift_50, tdrift_10

# drift times helper
def _percent_level_value(peak_val: float, level: float) -> float:
    """Helper: amplitude corresponding to a given fraction of the peak."""
    return peak_val * level

# drift times levels (another version?)
def compute_tdrift_levels(wf: np.ndarray,tp0_idx: int,peak_idx: int,levels=(0.10, 0.50, 0.999)):
    """
    Compute drift times from tp0 until given fractions of the peak.

    Parameters
    ----------
    wf : 1D np.ndarray
        Baseline-subtracted waveform.
    tp0_idx : int
        Index of tp0 (first rising sample; provided in metadata).
    peak_idx : int
        Index of the waveform maximum.
    levels : iterable of float
        Fractions of the peak amplitude (e.g., 0.1, 0.5, 0.999).

    Returns
    -------
    dict
        Keys 'tdrift10', 'tdrift50', 'tdrift99' (in samples).
    """
    segment = wf[tp0_idx:peak_idx + 1]
    peak_val = float(wf[peak_idx])
    results = {}
    for lvl in levels:
        target = _percent_level_value(peak_val, lvl)
        rel_idxs = np.where(segment >= target)[0]
        if len(rel_idxs) == 0:
            dt = np.nan
        else:
            dt = int(rel_idxs[0])  
        if np.isclose(lvl, 0.10):
            results["tdrift10"] = dt
        elif np.isclose(lvl, 0.50):
            results["tdrift50"] = dt
        else:  
            results["tdrift99"] = dt
    return results

# ------------------------------------------------------------
# 6. Time to Peak
# ------------------------------------------------------------
def compute_time_to_peak(wf, tp0):
    peak_index = np.argmax(wf)
    return peak_index - tp0