import numpy as np
from scipy.fft import rfft, rfftfreq


# ------------------------------------------------------------
# 1. Compute one-sided frequency spectrum
# ------------------------------------------------------------
def compute_frequency_spectrum(waveform, sample_spacing=1.0):
    y = np.asarray(waveform, dtype=float)
    N = len(y)

    yf = rfft(y)
    freqs = rfftfreq(N, d=sample_spacing)
    amplitude = np.abs(yf) * 2 / N

    return freqs, amplitude


# ------------------------------------------------------------
# 2. Peak frequency
# ------------------------------------------------------------
def compute_peak_frequency(waveform, sample_spacing=1.0):
    freqs, amp = compute_frequency_spectrum(waveform, sample_spacing)

    if len(amp) <= 1:
        return np.nan

    # skip DC bin at index 0
    peak_idx = np.argmax(amp[1:]) + 1
    return float(freqs[peak_idx])


# ------------------------------------------------------------
# 3. Spectral centroid
# ------------------------------------------------------------
def compute_spectral_centroid(waveform, sample_spacing=1.0):
    freqs, amp = compute_frequency_spectrum(waveform, sample_spacing)

    total_amp = np.sum(amp)
    if total_amp == 0:
        return 0.0

    centroid = np.sum(freqs * amp) / total_amp
    return float(centroid)
