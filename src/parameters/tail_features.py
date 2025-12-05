import numpy as np
from ..utils.transforms import pole_zero_correction, estimate_baseline


# ------------------------------------------------------------
# 1. Tail Charge Difference
# ------------------------------------------------------------
def compute_tail_charge_diff(waveform, energy, use_pz=True):
    """
    Normalized late-vs-early charge difference.

    Early window: 0.5–1.5 µs after peak
    Late window : 2.0–3.0 µs after peak

    tail_charge_diff = (late_area - early_area) / energy
    """

    # Optional PZ correction
    if use_pz:
        wf_pz, _ = pole_zero_correction(waveform)
    else:
        wf_pz = np.asarray(waveform, dtype=float)

    # Find peak index
    peak_index = np.argmax(wf_pz)

    S = 50  # samples per microsecond

    # Early window
    e_start = peak_index + int(0.5 * S)
    e_end   = peak_index + int(1.5 * S)

    # Late window
    l_start = peak_index + int(2.0 * S)
    l_end   = peak_index + int(3.0 * S)

    # Bounds check
    if l_end >= len(wf_pz) or e_start >= e_end:
        return np.nan

    early_area = float(np.sum(wf_pz[e_start:e_end]))
    late_area  = float(np.sum(wf_pz[l_start:l_end]))

    diff = late_area - early_area
    return diff / energy if energy > 0 else np.nan



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