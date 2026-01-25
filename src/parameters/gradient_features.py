import numpy as np
from src.utils.transforms import compute_gradient, pole_zero_correction, estimate_baseline
from scipy.stats import kurtosis, skew
from scipy.signal import savgol_filter

# ------------------------------------------------------------
# 1. Peak Count in Gradient
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 2. Gradient Baseline Noise
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 3. Current Kurtosis
# ------------------------------------------------------------
def compute_current_kurtosis(waveform, tp0_index, S=100):
    """
    Computes the Kurtosis of the Current Waveform during the rise.
    """    
    wf_smooth = savgol_filter(waveform, window_length=15, polyorder=3)
    current_waveform = compute_gradient(wf_smooth)
    # Extract current pulse from tp0 to peak + 2 microseconds
    end = min(len(current_waveform), tp0_index + int(2.0 * S)) 
    current_pulse = current_waveform[tp0_index : end]
    current_kurtosis = kurtosis(current_pulse, bias=False)
    return current_kurtosis

# ------------------------------------------------------------
# 4. Current Skewness
# ------------------------------------------------------------
def compute_current_skewness(waveform, tp0_index, S=100):
    """
    Computes the Skewness of the Current Waveform during the rise.
    """    
    wf_smooth = savgol_filter(waveform, window_length=15, polyorder=3)
    current_waveform = compute_gradient(wf_smooth)
    # Extract current pulse from tp0 to peak + 2 microseconds
    end = min(len(current_waveform), tp0_index + int(2.0 * S)) 
    current_pulse = current_waveform[tp0_index : end]
    current_skewness = skew(current_pulse, bias=False)
    return current_skewness

# ------------------------------------------------------------
# 5. Current Width
# ------------------------------------------------------------
def compute_current_width(waveform, tp0, S=100):
    """
    Returns the Full Width at Half Maximum (FWHM) of the PZ-corrected current.
    Uses Linear Interpolation via np.interp for sub-sample precision.
    """
    wf = np.asarray(waveform, dtype=float)
    
    # 1. PZ Correction 
    base, _ = estimate_baseline(wf)
    wf_pz, _ = pole_zero_correction(wf - base, use_pz=True)
    
    # 2. Compute Current (Derivative), smooth first to ensure monotonicity
    wf_smooth = savgol_filter(wf_pz, window_length=15, polyorder=3)
    current = compute_gradient(wf_smooth)
    
    # 3. Focus on the Rise Region
    end = min(len(current), tp0 + int(2.0 * S)) 
    pulse_slice = current[tp0:end]
    
    # 4. Find Max Amplitude
    if len(pulse_slice) < 3: return 0.0
    
    max_idx = np.argmax(pulse_slice)
    max_val = pulse_slice[max_idx]

    if max_val <= 0: return 0.0
    
    half_max = max_val / 2.0

    # 5. Split and Interpolate
    time_axis = np.arange(len(pulse_slice))
    
    # We go up to max_idx+1 so the peak is the last point of the left side.
    left_y = pulse_slice[:max_idx+1]
    left_x = time_axis[:max_idx+1]
    
    right_y = pulse_slice[max_idx:]
    right_x = time_axis[max_idx:]
    
    # Left Crossing (Rising Edge)
    # Note: If noise makes the rise non-monotonic, np.interp picks the first occurrence.
    if len(left_y) < 2: return 0.0
    t_left = np.interp(half_max, left_y, left_x)
    
    # Right Crossing (Falling Edge)
    if len(right_y) < 2: return 0.0
    t_right = np.interp(half_max, right_y[::-1], right_x[::-1])
    
    # 6. Calculate Width
    width_samples = t_right - t_left 
    
    if width_samples < 0: return 0.0
    
    return width_samples / float(S)