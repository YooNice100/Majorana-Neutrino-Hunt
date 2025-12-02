import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq


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
# Pole–Zero Correction (OPTIONAL — disabled by default)
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
