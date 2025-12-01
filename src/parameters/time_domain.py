import numpy as np
from .utils import estimate_baseline
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



def compute_drift_times(waveform, tp0, step=0.1):
    """
    Computes drift-time parameters by interpolating the rising edge.

    Returns the indices where the rising waveform reaches:
        - 10 percent of peak
        - 50 percent of peak
        - 99.9 percent of peak (tdrift)

    Parameters
    ----------
    waveform : array-like
        Full raw waveform.
    tp0 : int
        Estimated rising edge start index.
    step : float
        Interpolation step size (default 0.1).

    Returns
    -------
    tdrift_999 : int
        Index where waveform reaches 99.9 percent of peak.
    tdrift_50 : int
        Index where waveform reaches 50 percent of peak.
    tdrift_10 : int
        Index where waveform reaches 10 percent of peak.
    """
    y = np.asarray(waveform, dtype=float)
    peak_idx = int(np.argmax(y))

    rise = y[tp0 : peak_idx + 1]
    t = np.arange(len(rise))

    interp = interp1d(t, rise, kind="linear")
    t_new = np.arange(0, len(t) - 1, step)
    y_new = interp(t_new)

    peak_val = np.max(y_new)

    tdrift_999 = int(np.where(y_new >= 0.999 * peak_val)[0][0])
    tdrift_50  = int(np.where(y_new >= 0.5   * peak_val)[0][0])
    tdrift_10  = int(np.where(y_new >= 0.1   * peak_val)[0][0])

    return tdrift_999, tdrift_50, tdrift_10