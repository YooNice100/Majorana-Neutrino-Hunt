import numpy as np
from ..utils.transforms import estimate_baseline 

def compute_PPR(waveform, n_plateau=300):
    """
    Peak Plateau Ratio (PPR)

    Measures how much the waveform "flattens" toward the end of the trace.
    Defined as:

        PPR = (mean of last n_plateau samples) / (peak height)

    Parameters
    ----------
    waveform : array-like
        The raw waveform.
    n_plateau : int
        Number of samples to use at the end for averaging the plateau.

    Returns
    -------
    float
        The Peak Plateau Ratio. NaN if peak is zero.
    """
    y = np.asarray(waveform, dtype=float)

    peak_val = float(np.max(y))
    if peak_val <= 0:
        return np.nan  # avoid division by zero or negative peak

    # Average of last N samples (plateau region)
    plateau = float(np.mean(y[-n_plateau:]))

    return plateau / peak_val
