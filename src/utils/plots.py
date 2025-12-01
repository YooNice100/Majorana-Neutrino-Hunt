import numpy as np
import matplotlib.pyplot as plt

# Import feature functions
from .parameters.tail_features import compute_LQ80, compute_ND80
from .parameters.time_domain import (
    compute_peak_width_25_75,
    compute_energy_duration,
    compute_drift_times,
)
from .parameters.frequency_domain import (
    compute_frequency_spectrum,
    compute_peak_frequency,
    compute_spectral_centroid,
)
from .parameters.misc import compute_PPR
from .transforms import pole_zero_correction
from .utils import estimate_baseline


# ------------------------------------------------------------
# Helper: save or show
# ------------------------------------------------------------
def _finalize(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ------------------------------------------------------------
# 1. LQ80 — waveform-level plot
# ------------------------------------------------------------
def plot_LQ80_waveform(waveform, save_path=None):
    t = np.arange(len(waveform))
    waveform_pz = pole_zero_correction(waveform)[0]

    baseline, _ = estimate_baseline(waveform)
    peak = np.max(waveform)
    target80 = baseline + 0.8 * (peak - baseline)
    i80 = np.where(waveform >= target80)[0][0]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, waveform, label="Original", color="blue")
    ax.plot(t, waveform_pz, label="PZ-corrected", color="orange")
    ax.axhline(target80, linestyle="--", color="red", label="80% level")
    ax.axvline(i80, linestyle="--", color="black", label="i80")

    ax.fill_between(
        t[i80:], waveform[i80:], waveform_pz[i80:], color="purple", alpha=0.3
    )

    ax.set_title("LQ80 Visualization")
    ax.set_xlabel("Sample")
    ax.set_ylabel("ADC")
    ax.legend()
    return _finalize(fig, save_path)


# ------------------------------------------------------------
# 2. ND80 — waveform-level plot
# ------------------------------------------------------------
def plot_ND80_waveform(waveform, save_path=None):
    t = np.arange(len(waveform))
    depth_abs, idx_notch, depth_norm = compute_ND80(waveform)

    baseline, _ = estimate_baseline(waveform)
    peak = np.max(waveform)
    level80 = baseline + 0.8 * (peak - baseline)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, waveform, color="blue", label="Waveform")
    ax.axhline(level80, linestyle="--", color="red", label="80% level")

    if idx_notch is not None:
        ax.scatter([idx_notch], [waveform[idx_notch]], color="black", label="Max notch")

    ax.set_title(f"ND80 abs={depth_abs:.2f}, norm={depth_norm:.4f}")
    ax.set_xlabel("Sample")
    ax.set_ylabel("ADC")
    ax.legend()
    return _finalize(fig, save_path)


# ------------------------------------------------------------
# 3. Peak width 25–75%
# ------------------------------------------------------------
def plot_peak_width_waveform(waveform, save_path=None):
    t = np.arange(len(waveform))
    width, left_idx, right_idx = compute_peak_width_25_75(waveform)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, waveform, label="Waveform")

    if left_idx:
        ax.axvline(left_idx, linestyle="--", color="orange", label="25% crossing")
    if right_idx:
        ax.axvline(right_idx, linestyle="--", color="green", label="75% crossing")

    ax.set_title(f"25–75% Width = {width:.2f}")
    ax.set_xlabel("Sample")
    ax.set_ylabel("ADC")
    ax.legend()
    return _finalize(fig, save_path)


# ------------------------------------------------------------
# 4. Drift times (10%, 50%, 99.9%)
# ------------------------------------------------------------
def plot_drift_times_waveform(waveform, tp0, save_path=None):
    t = np.arange(len(waveform))
    t999, t50, t10 = compute_drift_times(waveform, tp0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, waveform, label="Waveform")

    ax.axvline(tp0, color="grey", linestyle="--", label="tp0")
    ax.axvline(t10, color="blue", linestyle="--", label="10%")
    ax.axvline(t50, color="green", linestyle="--", label="50%")
    ax.axvline(t999, color="red", linestyle="--", label="99.9%")

    ax.set_title("Drift-Time Features")
    ax.set_xlabel("Sample")
    ax.set_ylabel("ADC")
    ax.legend()
    return _finalize(fig, save_path)


# ------------------------------------------------------------
# 5. FFT — linear scale
# ------------------------------------------------------------
def plot_fft_linear(waveform, save_path=None):
    xf, amp = compute_frequency_spectrum(waveform)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xf, amp, color="purple")
    ax.set_title("FFT Amplitude Spectrum")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amplitude")
    return _finalize(fig, save_path)


# ------------------------------------------------------------
# 6. FFT — log-log scale
# ------------------------------------------------------------
def plot_fft_log(waveform, save_path=None):
    xf, amp = compute_frequency_spectrum(waveform)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.loglog(xf[1:], amp[1:], color="purple")  # skip 0 Hz for log
    ax.set_title("FFT Spectrum (Log-Log)")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amplitude")
    return _finalize(fig, save_path)


# ------------------------------------------------------------
# 7. Distribution plots (generic helper)
# ------------------------------------------------------------
def _hist_plot(data_sse, data_mse, title, xlabel, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(data_sse, bins=60, color="blue", alpha=0.6, label="SSE")
    ax.hist(data_mse, bins=60, color="red", alpha=0.6, label="MSE")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finalize(fig, save_path)


# ------------------------------------------------------------
# 8. Histograms for all features
# ------------------------------------------------------------
def plot_hist_LQ80(LQ80_sse, LQ80_mse, save_path=None):
    return _hist_plot(LQ80_sse, LQ80_mse, "LQ80 Distribution", "LQ80", save_path)


def plot_hist_ND80(ND80_sse, ND80_mse, save_path=None):
    return _hist_plot(ND80_sse, ND80_mse, "ND80 Distribution", "ND80 (normalized)", save_path)


def plot_hist_peak_width(width_sse, width_mse, save_path=None):
    return _hist_plot(width_sse, width_mse, "Peak Width 25–75%", "Width (samples)", save_path)


def plot_hist_energy_duration(dur_sse, dur_mse, save_path=None):
    return _hist_plot(dur_sse, dur_mse, "Energy Duration 90%", "Duration (samples)", save_path)


def plot_hist_drift_times(drift_sse, drift_mse, save_path=None):
    return _hist_plot(drift_sse, drift_mse, "Drift-Time Distribution", "Drift-Time Index", save_path)


def plot_hist_spectral_centroid(sc_sse, sc_mse, save_path=None):
    return _hist_plot(sc_sse, sc_mse, "Spectral Centroid", "Centroid (freq units)", save_path)


def plot_hist_peak_frequency(pf_sse, pf_mse, save_path=None):
    return _hist_plot(pf_sse, pf_mse, "Peak Frequency", "Frequency", save_path)


def plot_hist_PPR(ppr_sse, ppr_mse, save_path=None):
    return _hist_plot(ppr_sse, ppr_mse, "Peak Plateau Ratio (PPR)", "PPR", save_path)
