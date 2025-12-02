import numpy as np
from ..utils.transforms import pole_zero_correction, estimate_baseline




def compute_tail_charge_diff(waveform, energy, peak_index):
    """
    Energy-normalized late charge residual.

        tail_charge_diff = (late_area - early_area) / energy

    Early window: 0.5 to 1.5 microseconds after peak  
    Late window : 2.0 to 3.0 microseconds after peak  

    Parameters
    ----------
    waveform : array-like
        Raw waveform.
    energy : float
        DAQ energy proxy.
    peak_index : int
        Index of global maximum.

    Returns
    -------
    float
        The normalized late-vs-early charge difference.
    """
    wf_pz, _ = pole_zero_correction(waveform)

    S = 50  # samples per microsecond

    # Early window
    e_start = peak_index + int(0.5 * S)
    e_end   = peak_index + int(1.5 * S)

    # Late window
    l_start = peak_index + int(2.0 * S)
    l_end   = peak_index + int(3.0 * S)

    # Check bounds
    if l_end >= len(wf_pz) or e_start >= e_end:
        return np.nan

    early_area = float(np.sum(wf_pz[e_start:e_end]))
    late_area  = float(np.sum(wf_pz[l_start:l_end]))

    diff = late_area - early_area

    return diff / energy if energy > 0 else np.nan



def compute_LQ80(waveform, waveform_pz):
    """
    Late Charge 80:
    LQ80 = area(original) - area(corrected),
    integrated from the 80 percent crossing to the end.
    """
    y = np.asarray(waveform, dtype=float)
    yc = np.asarray(waveform_pz, dtype=float)
    t = np.arange(len(y), dtype=float)

    baseline, _ = estimate_baseline(y, 200)

    peak = float(np.max(y))
    target80 = baseline + 0.80 * (peak - baseline)

    idxs = np.where(y >= target80)[0]
    if len(idxs) == 0:
        return float("nan")

    i80 = int(idxs[0])

    area_raw = float(np.trapezoid(y[i80:], t[i80:]))
    area_corr = float(np.trapezoid(yc[i80:], t[i80:]))

    return area_raw - area_corr



def compute_ND80(waveform, n_pre=200):
    """
    Notch Depth 80:
    Maximum dip below 80 percent level
    between the 80 percent crossing and the peak.
    """
    y = np.asarray(waveform, dtype=float)

    baseline, _ = estimate_baseline(y, n_samples=n_pre)
    peak_idx = int(np.argmax(y))
    peak_val = float(y[peak_idx])
    amp = peak_val - baseline

    if amp <= 0:
        return float("nan"), None, float("nan")

    level80 = baseline + 0.80 * amp

    above = np.where(y >= level80)[0]
    if len(above) == 0:
        return float("nan"), None, float("nan")

    i80 = int(above[0])

    if i80 >= peak_idx:
        return 0.0, peak_idx, 0.0

    seg = y[i80:peak_idx + 1]
    depth_vec = level80 - seg
    depth_vec[depth_vec < 0] = 0

    depth_abs = float(np.max(depth_vec))
    idx_notch = int(i80 + np.argmax(depth_vec))
    depth_norm = float(depth_abs / amp) if amp > 0 else float("nan")

    return depth_abs, idx_notch, depth_norm
