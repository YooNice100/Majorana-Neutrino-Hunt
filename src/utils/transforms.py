import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d
from scipy.signal import windows

# ------------------------------------------------------------
# Baseline estimation
# ------------------------------------------------------------
def estimate_baseline(y, n_samples=200):
    """
    Returns baseline (mean, std) from the first n_samples.
    """
    y0 = np.asarray(y, dtype=float)[:n_samples]
    return float(np.mean(y0)), float(np.std(y0))



# ------------------------------------------------------------
# Double exponential tail model
# ------------------------------------------------------------
def exponential(t, a, tau1, b, tau2):
    """
    Double exponential decay model of the form:
        a * exp(-t/tau1) + b * exp(-t/tau2)

    This models the long falling edge of HPGe waveforms.
    """
    return a * np.exp(-t / tau1) + b * np.exp(-t / tau2)



# ------------------------------------------------------------
# Pole–Zero Correction 
# ------------------------------------------------------------
def pole_zero_correction(waveform, use_pz=True, min_tail_len=10):
    """
    Applies pole-zero correction to a given waveform.

    Returns:
        waveform_pz (np.array): The PZ-corrected full waveform.
        waveform_tail_corrected (np.array): The PZ-corrected tail portion of the waveform (from t98 onward).
    """
    waveform = np.asarray(waveform, dtype=float)
    if not use_pz:
        return waveform.copy(), waveform.copy()

    peak_value = float(np.max(waveform))
    if (not np.isfinite(peak_value)) or peak_value <= 0:
        return waveform.copy(), waveform.copy()

    # Isolate the tail (starting at 98% of the peak)
    idx98 = np.where(waveform >= 0.98 * peak_value)[0]
    if len(idx98) == 0:
        return waveform.copy(), waveform.copy()
    t98 = int(idx98[0])

    tail_values = waveform[t98:]
    tail_time = np.arange(tail_values.size, dtype=float)

    # If the tail is too short, we cannot fit 4 parameters
    if tail_values.size < min_tail_len:
        return waveform.copy(), tail_values.copy()

    # Initial guess
    p0 = [tail_values[0], 300.0, tail_values[0] * 0.1, 1500.0]

    try:
        params, _ = curve_fit(
            exponential,
            tail_time,
            tail_values,
            p0=p0,
            maxfev=20000
        )
    except (RuntimeError, ValueError, TypeError):
        return waveform.copy(), tail_values.copy()

    # Calculate the correction factor and apply it
    f_decay = exponential(tail_time, *params)

    # Estimate the initial value of the tail from a few samples near t98
    end = min(t98 + 5, len(waveform))
    f_t0 = float(np.mean(waveform[t98:end]))

    eps = 1e-12
    f_pz = f_t0 / (f_decay + eps)

    waveform_tail_corrected = tail_values * f_pz
    waveform_pz = waveform.copy()
    waveform_pz[t98:] = waveform_tail_corrected

    return waveform_pz, waveform_tail_corrected




    
# ------------------------------------------------------------
# Frequency Spectrum
# ------------------------------------------------------------
def compute_frequency_spectrum(waveform, sample_spacing=1.0):
    """
    Computes one-sided FFT amplitude spectrum of the waveform.

    Parameters
    ----------
    waveform : array-like
        The signal to transform.
    sample_spacing : float
        Time step between samples.

    Returns
    -------
    xf : np.ndarray
        Frequencies.
    amplitude : np.ndarray
        Real amplitude spectrum.
    """
    wf = np.asarray(waveform, dtype=float)
    N = len(wf)

    yf = rfft(wf)
    xf = rfftfreq(N, d=sample_spacing)

    amplitude = np.abs(yf) * 2.0 / N
    return xf, amplitude

# ------------------------------------------------------------
# Gradient
# ------------------------------------------------------------
def compute_gradient(wf: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Compute numerical derivative of a waveform.

    Parameters
    ----------
    wf : 1D np.ndarray
        Input waveform.
    dx : float
        Sample spacing (default 1.0).

    Returns
    -------
    np.ndarray
        Gradient of wf with respect to x.
    """
    wf = np.asarray(wf, dtype=float)
    return np.gradient(wf, dx)

# ------------------------------------------------------------
# RFFT Power Spectrum
# ------------------------------------------------------------
def compute_rfft_power_spectrum(
        raw_waveform,
        sample_spacing=1e-6,
        smoothing_sigma=0.2
    ):
    # Computes baseline-subtracted, smoothed, windowed FFT power spectrum.
    N = len(raw_waveform)
    # Baseline subtraction
    baseline, _ = estimate_baseline(raw_waveform)
    waveform = raw_waveform - baseline

    # Gaussian smoothing (avoid oversmoothing!)
    waveform = gaussian_filter1d(waveform, sigma=smoothing_sigma)

    # # Amplitude-preserving Hann window
    window = windows.hann(N)
    window_norm = window / (np.sum(window) / N)    # avg(window_norm) = 1
    waveform = waveform * window

    # FFT
    yf = rfft(waveform)
    xf = rfftfreq(N, d=sample_spacing)

    # Power spectrum (single-sided)
    # normalize by N to keep energy scale correct
    power_spectrum = (np.abs(yf) ** 2) / N

    return xf, power_spectrum

# ------------------------------------------------------------
# Find index of first peak after the steepest rise
# ------------------------------------------------------------
def peak_after_max_slope(wf, tp0, S=100, search_us=2.0):
    # Finds the peak index after the maximum slope following tp0.
    tp0 = int(tp0)
    start = max(tp0, 1)
    end = min(len(wf) - 1, tp0 + int(search_us * S))

    if end <= start:
        return None

    deriv = np.diff(wf)
    max_slope_idx = start + np.argmax(deriv[start:end])

    # search a short window after slope
    peak_start = max_slope_idx
    # 2.0 microseconds after max slope
    peak_end = min(len(wf), max_slope_idx + int(2.0 * S))

    return peak_start + int(np.argmax(wf[peak_start:peak_end]))