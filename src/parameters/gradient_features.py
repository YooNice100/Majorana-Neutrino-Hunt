import numpy as np
from .transforms import compute_gradient, pole_zero_correction, estimate_baseline
from scipy.stats import kurtosis, skew
from scipy.signal import savgol_filter, find_peaks

# ------------------------------------------------------------
# 1. Peak Count in Gradient
# ------------------------------------------------------------
def compute_peak_count(wf: np.ndarray,
                       dx: float = 1.0,
                       threshold_frac: float = 0.1,
                       min_separation: int = 5) -> int:

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

    wf_smooth = savgol_filter(waveform, window_length=15, polyorder=3)
    current_waveform = compute_gradient(wf_smooth)

    end = min(len(current_waveform), tp0_index + int(2.0 * S))
    current_pulse = current_waveform[tp0_index:end]

    return kurtosis(current_pulse, bias=False)


# ------------------------------------------------------------
# 4. Current Skewness
# ------------------------------------------------------------
def compute_current_skewness(waveform, tp0_index, S=100):

    wf_smooth = savgol_filter(waveform, window_length=15, polyorder=3)
    current_waveform = compute_gradient(wf_smooth)

    end = min(len(current_waveform), tp0_index + int(2.0 * S))
    current_pulse = current_waveform[tp0_index:end]

    return skew(current_pulse, bias=False)


# ------------------------------------------------------------
# 5. Current Width
# ------------------------------------------------------------
def compute_current_width(waveform, tp0, S=100):

    wf = np.asarray(waveform, dtype=float)

    base, _ = estimate_baseline(wf)
    wf_pz, _ = pole_zero_correction(wf - base, use_pz=True)

    wf_smooth = savgol_filter(wf_pz, window_length=15, polyorder=3)
    current = compute_gradient(wf_smooth)

    end = min(len(current), tp0 + int(2.0 * S))
    pulse_slice = current[tp0:end]

    if len(pulse_slice) < 3:
        return 0.0

    max_idx = np.argmax(pulse_slice)
    max_val = pulse_slice[max_idx]

    if max_val <= 0:
        return 0.0

    half_max = max_val / 2.0

    time_axis = np.arange(len(pulse_slice))

    left_y = pulse_slice[:max_idx+1]
    left_x = time_axis[:max_idx+1]

    right_y = pulse_slice[max_idx:]
    right_x = time_axis[max_idx:]

    if len(left_y) < 2:
        return 0.0
    t_left = np.interp(half_max, left_y, left_x)

    if len(right_y) < 2:
        return 0.0
    t_right = np.interp(half_max, right_y[::-1], right_x[::-1])

    width_samples = t_right - t_left

    if width_samples < 0:
        return 0.0

    return width_samples / float(S)


# ------------------------------------------------------------
# 6. Internal gradient lobe calculation
# ------------------------------------------------------------
def _compute_gradient_lobes(waveform,
                            n_baseline=50,
                            sg_win=7,
                            sg_poly=2):

    w = np.asarray(waveform, dtype=float)

    if len(w) == 0:
        return np.nan, np.nan

    baseline = np.mean(w[:min(n_baseline, len(w))])
    w0 = w - baseline

    if len(w0) >= sg_win:
        x = savgol_filter(w0, sg_win, sg_poly, mode="interp")
    else:
        x = w0

    d = np.diff(x)

    if len(d) == 0:
        return np.nan, np.nan

    max_d = np.max(d)
    prominence = max_d * 0.05 if max_d > 0 else 0.0

    peaks, _ = find_peaks(d, prominence=prominence)

    if len(peaks) == 0:
        return np.nan, np.nan

    lobes = []

    for p in peaks:
        L = p
        while L > 0 and d[L] > 0:
            L -= 1

        R = p
        while R < len(d) - 1 and d[R] > 0:
            R += 1

        lobe = np.clip(d[L:R+1], 0, None)
        area = float(np.trapezoid(lobe))

        lobes.append((area, L, R))

    if not lobes:
        return np.nan, np.nan

    lobes.sort(reverse=True)

    area1, L1, R1 = lobes[0]
    width_main = float(R1 - L1 + 1)

    if len(lobes) > 1:
        area2 = lobes[1][0]
    else:
        area2 = 0.0

    grad_area_ratio = area1 / (area2 + 1e-9)

    return grad_area_ratio, width_main


# ------------------------------------------------------------
# 7. GradAreaRatio
# ------------------------------------------------------------
def compute_grad_area_ratio(waveform):
    ratio, _ = _compute_gradient_lobes(waveform)
    return ratio


# ------------------------------------------------------------
# 8. GradWidthMain
# ------------------------------------------------------------
def compute_grad_width_main(waveform):
    _, width = _compute_gradient_lobes(waveform)
    return width