import numpy as np
from scipy.fft import rfft, rfftfreq


# ------------------------------------------------------------
# 1. Compute one-sided frequency spectrum (real FFT)
# ------------------------------------------------------------
def compute_frequency_spectrum(waveform, sample_spacing=1.0):
    """
    Computes the real FFT (one-sided) amplitude spectrum.

    Parameters
    ----------
    waveform : array-like
        Raw waveform values.
    sample_spacing : float
        Time difference between samples.

    Returns
    -------
    freqs : np.ndarray
        Frequency bins corresponding to FFT components.
    amplitude : np.ndarray
        Magnitude of frequency components (normalized).
    """
    y = np.asarray(waveform, dtype=float)
    N = len(y)

    yf = rfft(y)                         # FFT on real-valued signal
    freqs = rfftfreq(N, d=sample_spacing)

    # Normalize amplitude; factor 2 cancels missing negative-frequency half
    amplitude = np.abs(yf) * 2 / N

    return freqs, amplitude


# ------------------------------------------------------------
# 2. Peak Frequency (dominant FFT bin, ignoring DC component)
# ------------------------------------------------------------
def get_peak_frequency(waveform, sample_spacing=1.0):
    """
    Returns the frequency at which the FFT amplitude is largest,
    ignoring the DC (0 Hz) component.

    Parameters
    ----------
    waveform : array-like
        Raw waveform.
    sample_spacing : float
        Time spacing between samples.

    Returns
    -------
    float
        Dominant frequency (Hz or index units).
    """
    freqs, amp = compute_frequency_spectrum(waveform, sample_spacing)

    if len(amp) <= 1:
        return np.nan

    # Skip index 0 (DC component)
    peak_idx = np.argmax(amp[1:]) + 1

    return float(freqs[peak_idx])


# ------------------------------------------------------------
# 3. Spectral Centroid (frequency "center of mass")
# ------------------------------------------------------------
def compute_spectral_centroid(waveform, sample_spacing=1.0):
    """
    Computes spectral centroid:

        centroid = sum(freq * amplitude) / sum(amplitude)

    Represents the "balance point" of spectral energy.

    Parameters
    ----------
    waveform : array-like
        Raw waveform.
    sample_spacing : float
        Sample spacing for FFT.

    Returns
    -------
    float
        Spectral centroid frequency.
    """
    freqs, amp = compute_frequency_spectrum(waveform, sample_spacing)

    total_amp = np.sum(amp)
    if total_amp == 0:
        return 0.0

    centroid = np.sum(freqs * amp) / total_amp
    return float(centroid)
