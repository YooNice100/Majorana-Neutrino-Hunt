import numpy as np
from src.utils.transforms import compute_frequency_spectrum, compute_rfft_power_spectrum

# ------------------------------------------------------------
# 1. Peak frequency
# ------------------------------------------------------------
def compute_peak_frequency(waveform, sample_spacing=1.0):
    freqs, amp = compute_frequency_spectrum(waveform, sample_spacing)

    if len(amp) <= 1:
        return np.nan

    # skip DC bin at index 0
    peak_idx = np.argmax(amp[1:]) + 1
    return float(freqs[peak_idx])


# ------------------------------------------------------------
# 2. Spectral centroid
# ------------------------------------------------------------
def compute_spectral_centroid(waveform, sample_spacing=1.0):
    freqs, amp = compute_frequency_spectrum(waveform, sample_spacing)

    total_amp = np.sum(amp)
    if total_amp == 0:
        return 0.0

    centroid = np.sum(freqs * amp) / total_amp
    return float(centroid)


# ------------------------------------------------------------
# 3. HFER fourier sharpness
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

# ------------------------------------------------------------
# 4. Band Power Ratio
# ------------------------------------------------------------
def compute_band_power_ratio(wf: np.ndarray,fs: float,low_band=(0.0, 0.5e6),high_band=(0.5e6, 5.0e6)) -> float:
    """
    Band Power Ratio (BPR) = high-band power / low-band power.

    Parameters
    ----------
    wf : 1D np.ndarray
        Baseline-subtracted waveform.
    fs : float
        Sampling frequency in Hz (100e6 for this dataset).
    low_band : (float, float)
        [f_min, f_max] for the low-frequency band (Hz).
    high_band : (float, float)
        [f_min, f_max] for the high-frequency band (Hz).

    Returns
    -------
    float
        Band power ratio.
    """
    x = np.asarray(wf, dtype=float)
    n = x.size
    if n == 0:
        return np.nan

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.rfft(x)
    power = np.abs(fft_vals) ** 2

    # low band
    low_mask = (freqs >= low_band[0]) & (freqs < low_band[1])
    high_mask = (freqs >= high_band[0]) & (freqs < high_band[1])

    low_power = float(np.sum(power[low_mask]))
    high_power = float(np.sum(power[high_mask]))

    if low_power == 0:
        return np.nan

    return high_power / low_power

# ------------------------------------------------------------
# 5. Spectral centroid using power spectrum
# ------------------------------------------------------------
def compute_spectral_centroid_power(wf):
    xf, power_spectrum = compute_rfft_power_spectrum(wf)
    ps_sum = np.sum(power_spectrum)
    if ps_sum <= 0:
        return 0.0
    return np.sum(xf * power_spectrum) / ps_sum

# ------------------------------------------------------------
# 6. Total power using power spectrum
# ------------------------------------------------------------
def compute_total_power(wf):
    xf, power_spectrum = compute_rfft_power_spectrum(wf)
    total_power = np.sum(power_spectrum)
    return total_power