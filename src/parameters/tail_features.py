import numpy as np
from ..utils.transforms import pole_zero_correction, estimate_baseline


# ------------------------------------------------------------
# 1. Tail Slope (Normalized by Energy)
# ------------------------------------------------------------
def compute_tail_slope(wf, tp0, use_pz=True, S=100):
    """
    Calculates the Slope of the PZ tail, normalized by Energy.
    
    Returns:
    - slope_norm: Units of (1/us).
                  0.0  = Flat (Bulk)
                  < 0  = Drooping (Surface/Trapping)
    """
    wf_pz, _ = pole_zero_correction(wf, use_pz=use_pz)
    
    # 1. Define the 'Deep Tail' Window
    start_us = 4.0
    start_idx = int(tp0 + (start_us * S))
    end_idx = len(wf_pz) - 5
    
    # Safety Check
    if start_idx >= end_idx - 10:
        return 0.0
        
    # 2. Extract Data
    tail_slice = wf_pz[start_idx : end_idx]
    
    # 3. Get Energy (Peak Height) for Normalization
    # We look at the pulse from trigger onwards
    energy = np.max(wf_pz[tp0:])
    
    if energy < 10.0: return 0.0 # Avoid div/0 on noise
    
    # 4. Normalize the DATA, not the slope
    # This makes the fit result automatically normalized
    tail_norm = tail_slice / energy
    
    # 5. Linear Fit on Normalized Data
    # X-axis in microseconds
    time_axis = np.arange(len(tail_norm)) / float(S)
    
    # slope is now in units of [Normalized_ADC / us] -> [1/us]
    slope, intercept = np.polyfit(time_axis, tail_norm, 1)
    
    return slope

# ------------------------------------------------------------
# 2. LQ80
# ------------------------------------------------------------
def compute_LQ80(waveform):
    """
    Late Charge 80:
    Area difference between raw and PZ-corrected waveform
    starting at the 80 percent rising edge.
    """
    waveform_pz, _ = pole_zero_correction(waveform)
    y  = np.asarray(waveform, dtype=float)
    yc = np.asarray(waveform_pz, dtype=float)

    # Baseline
    baseline, _ = estimate_baseline(y)

    # Peak
    peak_val = float(np.max(y))
    target = baseline + 0.80 * (peak_val - baseline)

    # Rising-edge crossing
    idx = np.where(y >= target)[0]
    if len(idx) == 0:
        return np.nan

    i80 = int(idx[0])

    # Time index for integration
    t = np.arange(len(y), dtype=float)

    area_raw  = float(np.trapezoid(y[i80:],  t[i80:]))
    area_corr = float(np.trapezoid(yc[i80:], t[i80:]))

    return area_raw - area_corr



# ------------------------------------------------------------
# 3. ND80 (Notch Depth 80)
# ------------------------------------------------------------
def compute_ND80(waveform, n_pre=200):
    """
    ND80: Maximum dip below the 80 percent amplitude level,
    between the 80 percent crossing and the peak.

    Returns:
        depth_abs  : absolute depth
        idx_notch  : index of notch
        depth_norm : normalized by peak amplitude
    """

    y = np.asarray(waveform, dtype=float)

    baseline, _ = estimate_baseline(y, n_samples=n_pre)
    peak_idx = int(np.argmax(y))
    peak_val = float(y[peak_idx])
    amp = peak_val - baseline

    if amp <= 0:
        return np.nan, None, np.nan

    # 80 percent level
    level80 = baseline + 0.80 * amp

    # Find 80 percent crossing
    above = np.where(y >= level80)[0]
    if len(above) == 0:
        return np.nan, None, np.nan

    i80 = int(above[0])

    # If 80 percent crossing is after peak, no notch exists
    if i80 >= peak_idx:
        return 0.0, peak_idx, 0.0

    seg = y[i80: peak_idx + 1]

    # Depth below level80
    depth_vec = level80 - seg
    depth_vec[depth_vec < 0] = 0  # only dips count

    depth_abs = float(np.max(depth_vec))
    idx_notch = int(i80 + np.argmax(depth_vec))

    depth_norm = depth_abs / amp if amp > 0 else np.nan

    return depth_abs, idx_notch, depth_norm

# ------------------------------------------------------------
# 4. Tail Flattening Ratio (TFR)
# ------------------------------------------------------------
def compute_tfr(wf: np.ndarray,peak_idx: int,tau_samples: float,tail_len: int = 1000) -> float:
    """
    Tail Flattening Ratio (TFR).

    TFR = std(tail_raw) / std(tail_pz)

    Parameters
    ----------
    wf : 1D np.ndarray
        Raw waveform.
    peak_idx : int
        Index of the pulse peak.
    tau_samples : float
        Decay constant used for pole-zero correction (in samples).
    tail_len : int
        Number of samples in the tail region after the peak.

    Returns
    -------
    float
        Tail Flattening Ratio value.
    """
    wf = np.asarray(wf, dtype=float)

    # Safety on tail window
    start = peak_idx
    stop = min(peak_idx + tail_len, wf.size)
    if stop <= start + 5:
        return np.nan

    tail_raw = wf[start:stop]
    wf_pz = pole_zero_correction(wf, tau_samples)
    tail_pz = wf_pz[start:stop]

    std_raw = float(np.std(tail_raw))
    std_pz = float(np.std(tail_pz))

    if std_pz == 0:
        return np.nan

    return std_raw / std_pz

# ------------------------------------------------------------
# 5. Peak Plateau Ratio (PPR)
# ------------------------------------------------------------
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