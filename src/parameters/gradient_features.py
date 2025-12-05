import numpy as np
from src.utils.transforms import compute_gradient

def compute_peak_count(wf: np.ndarray,
                       dx: float = 1.0,
                       threshold_frac: float = 0.1,
                       min_separation: int = 5) -> int:
    """
    Count significant local maxima in the *gradient* of a waveform.

    Parameters
    ----------
    wf : 1D np.ndarray
        Input waveform (baseline-subtracted and optionally smoothed).
    dx : float
        Sample spacing used for the gradient.
    threshold_frac : float
        Fraction of the global max gradient used as a minimum peak height.
    min_separation : int
        Minimum distance (in samples) between distinct peaks.

    Returns
    -------
    int
        Number of gradient peaks above threshold.
    """
    grad = compute_gradient(wf, dx=dx)
    g = np.asarray(grad, dtype=float)
    if g.size < 3:
        return 0

    max_val = np.max(g)
    if max_val <= 0:
        return 0

    thr = threshold_frac * max_val
    peaks = []

    for i in range(1, g.size - 1):
        if g[i] > thr and g[i] >= g[i - 1] and g[i] > g[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_separation:
                peaks.append(i)

    return len(peaks)

def compute_gradient_baseline_noise(wf: np.ndarray,
                                    dx: float = 1.0,
                                    baseline_end: int = 200) -> float:
    """
    RMS of the gradient in the pre-pulse baseline region.

    Parameters
    ----------
    wf : 1D np.ndarray
        Input waveform.
    dx : float
        Sample spacing for gradient.
    baseline_end : int
        Index (exclusive) marking the end of the baseline region.

    Returns
    -------
    float
        Baseline gradient RMS.
    """
    grad = compute_gradient(wf, dx=dx)
    g = np.asarray(grad, dtype=float)
    end = min(baseline_end, g.size)
    if end <= 1:
        return np.nan
    return float(np.sqrt(np.mean(g[:end] ** 2)))

