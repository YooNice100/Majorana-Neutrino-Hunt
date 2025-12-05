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
    Applies pole–zero correction to the waveform tail.

    Parameters
    ----------
    waveform : np.ndarray
        Raw waveform.
    use_pz : bool
        If False, returns the waveform unchanged.
        If True, attempts exponential fitting and tail correction.

    Returns
    -------
    waveform_pz : np.ndarray
        Corrected waveform (or raw waveform if disabled/fitting failed).
    corrected_tail : np.ndarray
        The corrected tail region (or raw tail if not corrected).
    """

    # --------------------------------------------------------
    # If disabled → return original waveform immediately
    # --------------------------------------------------------
    y = np.asarray(waveform, dtype=float)
    if not use_pz:
        return y, y

    # --------------------------------------------------------
    # Identify 98 percent rise point (start of decay)
    # --------------------------------------------------------
    peak_value = np.max(y)
    t98_idx = np.where(y >= 0.98 * peak_value)[0]
    if len(t98_idx) == 0:
        return y, y
    t98 = int(t98_idx[0])

    # Tail region
    tail_values = y[t98:]
    tail_time = np.arange(len(tail_values))

    # --------------------------------------------------------
    # Fit a double exponential tail
    # Use tighter bounds to avoid overflow and unrealistic fits
    # --------------------------------------------------------
    try:
        params, _ = curve_fit(
            exponential,
            tail_time,
            tail_values,
            p0=[peak_value, 300.0, peak_value * 0.1, 1500.0],  # initial guesses
            bounds=(
                [0, 10, 0, 10],          # lower bounds
                [peak_value * 2, 5000, peak_value * 2, 5000]  # upper bounds
            ),
            maxfev=4000
        )

        # Decay model
        model_decay = exponential(tail_time, *params)

        # Reference point for normalization
        f_t0 = np.mean(tail_values[:5])
        f_pz = f_t0 / model_decay

        corrected_tail = tail_values * f_pz

        waveform_pz = y.copy()
        waveform_pz[t98:] = corrected_tail

        return waveform_pz, corrected_tail

    except Exception:
        # If fitting fails, return original
        return y, y


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