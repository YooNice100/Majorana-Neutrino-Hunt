# src/experiments/exp_all.py

import os
import numpy as np
from scipy.stats import ttest_ind

# Loaders and plotting
from src.utils.io import load_hdf5
from src.utils.stats import make_strict_masks
from src.utils.plots import (
    plot_hist_LQ80,
    plot_hist_ND80,
    plot_hist_peak_width,
    plot_hist_energy_duration,
    plot_hist_drift_times,
    plot_hist_peak_frequency,
    plot_hist_spectral_centroid,
    plot_hist_avse,
    plot_hist_hfer,
)

# Feature functions
from src.parameters.tail_features import compute_LQ80, compute_ND80, compute_tfr
from src.parameters.time_domain import (
    compute_peak_width_25_75,
    compute_energy_duration,
    estimate_tp0_threshold,
    compute_drift_times,
    compute_avse,
    compute_tdrift_levels
)


from src.parameters.frequency_domain import (
    compute_peak_frequency,
    compute_spectral_centroid,
    compute_hfer,
    
)

from src.parameters.gradient_features import (
    compute_peak_count,
    compute_gradient_baseline_noise,
)


# Import ONLY for LQ80
from src.utils.transforms import pole_zero_correction


DATA_PATH = "data/MJD_Train_2.hdf5"
OUT_DIR = "graphs"
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------------------------------
# Load data + masks
# -------------------------------------------------
print(f"Loading: {DATA_PATH}")
data = load_hdf5(DATA_PATH)

waveforms = data["raw_waveform"]
strict_sse, strict_mse = make_strict_masks(data)

print(f"Total events: {len(waveforms)}")
print(f"SSE count: {strict_sse.sum()}, MSE count: {strict_mse.sum()}")


# -------------------------------------------------
# Compute ALL features (no PZ except LQ80)
# -------------------------------------------------
LQ80 = []
ND80 = []
peak_width = []
energy_dur = []
drift_50 = []
peak_freq = []
spec_centroid = []

for wf in waveforms:

    # --- LQ80 only ---
    wf_pz, _ = pole_zero_correction(wf)
    LQ80.append(compute_LQ80(wf, wf_pz))

    # --- ND80 ---
    _, _, nd_norm = compute_ND80(wf)
    ND80.append(nd_norm)

    # --- Peak width ---
    width, _, _ = compute_peak_width_25_75(wf)
    peak_width.append(width)

    # --- Energy duration ---
    energy_dur.append(compute_energy_duration(wf))

    # --- Drift time ---
    tp0 = estimate_tp0_threshold(wf)
    if tp0 is None:
        drift_50.append(np.nan)
    else:
        _, t50, _ = compute_drift_times(wf, tp0)
        drift_50.append(t50)

    # --- Frequency domain ---
    peak_freq.append(compute_peak_frequency(wf))
    spec_centroid.append(compute_spectral_centroid(wf))


# Convert to arrays
LQ80 = np.array(LQ80)
ND80 = np.array(ND80)
peak_width = np.array(peak_width)
energy_dur = np.array(energy_dur)
drift_50 = np.array(drift_50)
peak_freq = np.array(peak_freq)
spec_centroid = np.array(spec_centroid)


# -------------------------------------------------
# Stats helper
# -------------------------------------------------
def clean(x):
    x = np.asarray(x)
    return x[np.isfinite(x)]

def ttest(name, sse_vals, mse_vals):
    xs = clean(sse_vals)
    xm = clean(mse_vals)
    if len(xs) == 0 or len(xm) == 0:
        print(f"{name}: insufficient data")
        return
    t, p = ttest_ind(xs, xm, equal_var=False)
    print(f"{name}: t = {t:.2f}, p = {p:.3e}")

# --- AvsE (vectorized) ---
avse, amp = compute_avse(data["raw_waveform"], data["energy_label"])  # returns (N,), (N,)

# Clean NaNs
def _clean(x):
    x = np.asarray(x)
    return x[np.isfinite(x)]

avse_sse = _clean(avse[strict_sse])
avse_mse = _clean(avse[strict_mse])

# --- hfer  ---
hfer_vals = []
for wf in waveforms:
    hfer_vals.append(compute_hfer(wf))  
hfer_vals = np.array(hfer_vals)

# -------------------------------------------------
# Summary
# -------------------------------------------------
print("\n=== SUMMARY OF ALL FEATURES ===")

ttest("LQ80", LQ80[strict_sse], LQ80[strict_mse])
ttest("ND80", ND80[strict_sse], ND80[strict_mse])
ttest("Peak Width", peak_width[strict_sse], peak_width[strict_mse])
ttest("Energy Duration", energy_dur[strict_sse], energy_dur[strict_mse])
ttest("Drift Time (50 percent)", drift_50[strict_sse], drift_50[strict_mse])
ttest("Peak Frequency", peak_freq[strict_sse], peak_freq[strict_mse])
ttest("Spectral Centroid", spec_centroid[strict_sse], spec_centroid[strict_mse])
ttest("AvsE", avse[strict_sse], avse[strict_mse])
ttest("HFER", hfer_vals[strict_sse], hfer_vals[strict_mse])

# -------------------------------------------------
# Save all histograms
# -------------------------------------------------
plot_hist_LQ80(LQ80[strict_sse], LQ80[strict_mse], f"{OUT_DIR}/LQ80_hist.png")
plot_hist_ND80(ND80[strict_sse], ND80[strict_mse], f"{OUT_DIR}/ND80_hist.png")
plot_hist_peak_width(peak_width[strict_sse], peak_width[strict_mse], f"{OUT_DIR}/peak_width_hist.png")
plot_hist_energy_duration(energy_dur[strict_sse], energy_dur[strict_mse], f"{OUT_DIR}/energy_duration_hist.png")
plot_hist_drift_times(drift_50[strict_sse], drift_50[strict_mse], f"{OUT_DIR}/drift_times_hist.png")
plot_hist_peak_frequency(peak_freq[strict_sse], peak_freq[strict_mse], f"{OUT_DIR}/peak_freq_hist.png")
plot_hist_spectral_centroid(spec_centroid[strict_sse], spec_centroid[strict_mse], f"{OUT_DIR}/spectral_centroid_hist.png")
plot_hist_avse(avse[strict_sse], avse[strict_mse], f"{OUT_DIR}/avse_hist.png")
plot_hist_hfer(hfer_vals[strict_sse], hfer_vals[strict_mse], f"{OUT_DIR}/hfer_hist.png")

print("\nSaved ALL plots to graphs/ folder.")
print("Done.\n")
