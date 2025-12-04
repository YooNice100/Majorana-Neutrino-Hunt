import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Import feature functions
# ------------------------------------------------------------
from ..parameters.tail_features import compute_LQ80, compute_ND80
from ..parameters.time_domain import (
    compute_peak_width_25_75,
    compute_energy_duration,
    compute_drift_times,
)
from ..parameters.frequency_domain import (
    compute_frequency_spectrum,
    compute_peak_frequency,
    compute_spectral_centroid,
)
from ..parameters.misc import compute_PPR

from ..utils.transforms import pole_zero_correction, estimate_baseline


# ------------------------------------------------------------
# Helper for saving
# ------------------------------------------------------------
def _finalize(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ------------------------------------------------------------
# LQ80 waveform plot
# ------------------------------------------------------------
def plot_LQ80_waveform(waveform, save_path=None):
    t = np.arange(len(waveform))
    waveform_pz = pole_zero_correction(waveform)[0]

    baseline, _ = estimate_baseline(waveform)
    peak = np.max(waveform)
    target80 = baseline + 0.8 * (peak - baseline)
    i80 = np.where(waveform >= target80)[0][0]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, waveform, label="Original")
    ax.plot(t, waveform_pz, label="PZ-corrected")
    ax.axhline(target80, linestyle="--", color="red", label="80 percent level")
    ax.axvline(i80, linestyle="--", color="black", label="i80")

    ax.fill_between(t[i80:], waveform[i80:], waveform_pz[i80:], color="purple", alpha=0.3)

    ax.set_title("LQ80 Visualization")
    ax.set_xlabel("Sample")
    ax.set_ylabel("ADC")
    ax.legend()
    return _finalize(fig, save_path)


# ------------------------------------------------------------
# ND80 waveform plot
# ------------------------------------------------------------
def plot_ND80_waveform(waveform, save_path=None):
    t = np.arange(len(waveform))
    depth_abs, idx_notch, depth_norm = compute_ND80(waveform)

    baseline, _ = estimate_baseline(waveform)
    peak = np.max(waveform)
    level80 = baseline + 0.8 * (peak - baseline)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, waveform, label="Waveform")
    ax.axhline(level80, linestyle="--", color="red", label="80 percent level")

    if idx_notch is not None:
        ax.scatter([idx_notch], [waveform[idx_notch]], color="black", label="Max notch")

    ax.set_title(f"ND80 abs={depth_abs:.2f}, norm={depth_norm:.4f}")
    ax.set_xlabel("Sample")
    ax.set_ylabel("ADC")
    ax.legend()
    return _finalize(fig, save_path)


# ------------------------------------------------------------
# 25–75 percent peak width plot
# ------------------------------------------------------------
def plot_peak_width_waveform(waveform, save_path=None):
    t = np.arange(len(waveform))
    width, left_idx, right_idx = compute_peak_width_25_75(waveform)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, waveform, label="Waveform")

    if left_idx is not None:
        ax.axvline(left_idx, linestyle="--", color="orange", label="25 percent")
    if right_idx is not None:
        ax.axvline(right_idx, linestyle="--", color="green", label="75 percent")

    ax.set_title(f"25–75 percent Width = {width:.2f}")
    ax.set_xlabel("Sample")
    ax.set_ylabel("ADC")
    ax.legend()
    return _finalize(fig, save_path)


# ------------------------------------------------------------
# Drift times (10 percent, 50 percent, 99.9 percent)
# ------------------------------------------------------------
def plot_drift_times_waveform(waveform, tp0, save_path=None):
    t = np.arange(len(waveform))
    t999, t50, t10 = compute_drift_times(waveform, tp0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, waveform, label="Waveform")

    ax.axvline(tp0, linestyle="--", color="gray", label="tp0")
    ax.axvline(t10, linestyle="--", color="blue", label="10 percent")
    ax.axvline(t50, linestyle="--", color="green", label="50 percent")
    ax.axvline(t999, linestyle="--", color="red", label="99.9 percent")

    ax.set_title("Drift-Time Features")
    ax.set_xlabel("Sample")
    ax.set_ylabel("ADC")
    ax.legend()
    return _finalize(fig, save_path)


# ------------------------------------------------------------
# FFT – linear
# ------------------------------------------------------------
def plot_fft_linear(waveform, save_path=None):
    xf, amp = compute_frequency_spectrum(waveform)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xf, amp)
    ax.set_title("FFT Amplitude Spectrum")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amplitude")
    return _finalize(fig, save_path)


# ------------------------------------------------------------
# FFT – log scale
# ------------------------------------------------------------
def plot_fft_log(waveform, save_path=None):
    xf, amp = compute_frequency_spectrum(waveform)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.loglog(xf[1:], amp[1:])  # skip 0 Hz
    ax.set_title("FFT Spectrum (Log–Log)")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amplitude")
    return _finalize(fig, save_path)


# ------------------------------------------------------------
# Simple histogram helper
# ------------------------------------------------------------
def _hist_plot(data_sse, data_mse, title, xlabel, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(data_sse, bins=60, alpha=0.6, label="SSE")
    ax.hist(data_mse, bins=60, alpha=0.6, label="MSE")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finalize(fig, save_path)


# ------------------------------------------------------------
# Histogram wrappers for each feature
# ------------------------------------------------------------
def plot_hist_LQ80(sse, mse, save_path=None):
    return _hist_plot(sse, mse, "LQ80 Distribution", "LQ80", save_path)


def plot_hist_ND80(sse, mse, save_path=None):
    return _hist_plot(sse, mse, "ND80 Distribution", "ND80", save_path)


def plot_hist_peak_width(sse, mse, save_path=None):
    return _hist_plot(sse, mse, "Peak Width 25–75 percent", "Width", save_path)


def plot_hist_energy_duration(sse, mse, save_path=None):
    return _hist_plot(sse, mse, "Energy Duration", "Samples", save_path)


def plot_hist_drift_times(sse, mse, save_path=None):
    return _hist_plot(sse, mse, "Drift-Time Distribution", "Drift-Time", save_path)


def plot_hist_spectral_centroid(sse, mse, save_path=None):
    return _hist_plot(sse, mse, "Spectral Centroid", "Freq Units", save_path)


def plot_hist_peak_frequency(sse, mse, save_path=None):
    return _hist_plot(sse, mse, "Peak Frequency", "Frequency", save_path)


def plot_hist_PPR(sse, mse, save_path=None):
    return _hist_plot(sse, mse, "Peak Plateau Ratio (PPR)", "PPR", save_path)

def plot_hist_avse(sse, mse, save_path=None):
    return _hist_plot(sse, mse, "AvsE Distribution", "AvsE", save_path)