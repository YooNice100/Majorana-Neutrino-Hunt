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


# ------------------------------------------------------------
# 4. HFER fourier sharpness
# ------------------------------------------------------------

def compute_hfer(waveform, frac_high=0.25, n_baseline=50):
    w = np.asarray(waveform, dtype=float)
    if w.size == 0:
        return np.nan

    b = np.mean(w[:min(n_baseline, len(w))])
    w0 = w - b
    peak = np.max(w0)
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    x = w0 / peak

    mag = np.abs(np.fft.rfft(x))
    if mag.size <= 1:
        return np.nan  
    mag_no_dc = mag[1:] 

    K = mag_no_dc.size
    k0 = int(max(0, np.floor((1.0 - frac_high) * K)))
    num = np.sum(mag_no_dc[k0:])
    den = np.sum(mag_no_dc)
    return float(num / den) if den > 0 else np.nan