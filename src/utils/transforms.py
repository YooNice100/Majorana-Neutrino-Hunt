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
def pole_zero_correction(waveform, use_pz=False):
    """
    Applies pole-zero correction to a given waveform.

    Args:
        raw waveform (np.array)
    Returns:
        waveform_pz (np.array): The PZ-corrected full waveform.
        waveform_tail_corrected (np.array): The PZ-corrected tail portion of the waveform.
    """
    waveform = np.asarray(waveform, dtype=float)
    if not use_pz:
        return waveform, waveform

    # Identify the peak values
    peak_value = np.max(waveform)
    # Isolate the tail (starting at 98% of the peak)
    t98 = np.where(waveform >= 0.98 * peak_value)[0][0] 
    # Generate the time index necessary for the fit 
    time_index = np.arange(0, len(waveform))
    tail_time = np.arange(0, time_index[-1] - t98 + 1)
    tail_values = waveform[t98:]

    # initial guess
    p0 = [tail_values[0], 300.0, tail_values[0] * 0.1, 1500.0]
    # Fit the parameters (with error handling)
    try:
        # Fit the decay model to the raw tail values
        params, params_cov = curve_fit(
            exponential, 
            tail_time, 
            tail_values,
            p0=p0, 
            maxfev=20000)
    except (RuntimeError, ValueError):
        # If the curve_fit fails to converge (e.g., due to noise or pile-up), 
        # return a copy of the original waveform to prevent the pipeline from crashing.
        return np.copy(waveform), waveform[t98:]

    # Calculate the correction factor and apply it
    f_decay = exponential(tail_time, *params)   
    # Estimate the initial value of the tail (f_t0) from the first few samples near t98
    f_t0 = np.mean(waveform[t98:t98+5])
    # Calculate the inverse correction factor (f_pz). 
    # This factor, when multiplied by the tail, flattens the exponential decay.    
    eps = 1e-12
    f_pz = f_t0 / (f_decay + eps)
  
    # Apply the correction
    waveform_tail_corrected = tail_values * f_pz
    # Create the final corrected waveform
    waveform_pz = np.copy(waveform)
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