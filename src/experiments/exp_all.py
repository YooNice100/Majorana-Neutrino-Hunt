# src/experiments/exp_all.py

import os
import numpy as np
from scipy.stats import ttest_ind

# Loaders and plotting
from src.utils.io import load_hdf5
from src.utils.stats import make_sse_mse_masks, _clean
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
    plot_hist_current_kurtosis,
    plot_hist_current_skewness,
    plot_hist_spectral_centroid_power,
    plot_hist_tail_charge_diff,
    plot_hist_time_to_peak,
    plot_hist_total_power
)

# Feature functions
from src.parameters.tail_features import (
    compute_LQ80, 
    compute_ND80, 
    compute_tfr,
    compute_tail_charge_diff
)
from src.parameters.time_domain import (
    compute_peak_width_25_75,
    compute_energy_duration,
    estimate_tp0_threshold,
    compute_drift_times,
    compute_avse,
    compute_tdrift_levels,
    compute_time_to_peak
)
from src.parameters.frequency_domain import (
    compute_peak_frequency,
    compute_spectral_centroid,
    compute_hfer,   
    compute_total_power,
    compute_spectral_centroid_power
)
from src.parameters.gradient_features import (
    compute_peak_count,
    compute_gradient_baseline_noise,
    compute_current_kurtosis,
    compute_current_skewness
)


DATA_PATH = "data/MJD_Train_2.hdf5"
OUT_DIR = "graphs"
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------------------------------
# Load data + masks
# -------------------------------------------------
print(f"Loading: {DATA_PATH}")
data = load_hdf5(DATA_PATH)

energy_label = data["energy_label"]
waveforms = data["raw_waveform"]
tp0_values = data["tp0"]
sse, mse = make_sse_mse_masks(data)

print(f"Total events: {len(waveforms)}")
print(f"SSE count: {sse.sum()}, MSE count: {mse.sum()}")


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
spec_centroid_power = []
tail_charge_diff = []
time_to_peak = []
total_power = []
current_skew = []
current_kurt = []

for i, wf in enumerate(waveforms):

    # --- LQ80 only ---
    LQ80.append(compute_LQ80(wf))

    # --- ND80 ---
    _, _, nd_norm = compute_ND80(wf)
    ND80.append(nd_norm)

    # --- Peak width ---
    width, _, _ = compute_peak_width_25_75(wf)
    peak_width.append(width)

    # --- Energy duration ---
    energy_dur.append(compute_energy_duration(wf))

    # --- True tp0 ---
    tp0_true = tp0_values[i]

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
    spec_centroid_power.append(compute_spectral_centroid_power(wf))
    total_power.append(compute_total_power(wf))

    # --- Tail charge difference ---
    energy = energy_label[i]
    tail_charge_diff.append(compute_tail_charge_diff(wf, energy))

    # --- Time to peak ---
    time_to_peak.append(compute_time_to_peak(wf, tp0_true))

    # --- Current kurtosis and skewness ---
    current_kurt.append(compute_current_kurtosis(wf, tp0_true))
    current_skew.append(compute_current_skewness(wf, tp0_true))



# Convert to arrays
LQ80 = np.array(LQ80)
ND80 = np.array(ND80)
peak_width = np.array(peak_width)
energy_dur = np.array(energy_dur)
drift_50 = np.array(drift_50)
peak_freq = np.array(peak_freq)
spec_centroid = np.array(spec_centroid)
spec_centroid_power = np.array(spec_centroid_power)
tail_charge_diff = np.array(tail_charge_diff)
time_to_peak = np.array(time_to_peak)
total_power = np.array(total_power)
current_kurt = np.array(current_kurt)
current_skew = np.array(current_skew)

# --- AvsE (vectorized) ---
avse, amp = compute_avse(data["raw_waveform"], data["energy_label"])  # returns (N,), (N,)

avse_sse = _clean(avse[sse])
avse_mse = _clean(avse[mse])

# --- hfer  ---
hfer_vals = []
for wf in waveforms:
    hfer_vals.append(compute_hfer(wf))  
hfer_vals = np.array(hfer_vals)

# -------------------------------------------------
# Stats helper
# -------------------------------------------------
def ttest(name, sse_vals, mse_vals):
    xs = _clean(sse_vals)
    xm = _clean(mse_vals)
    if len(xs) == 0 or len(xm) == 0:
        print(f"{name}: insufficient data")
        return
    t, p = ttest_ind(xs, xm, equal_var=False)
    print(f"{name}: t = {t:.2f}, p = {p:.3e}")


# -------------------------------------------------
# Summary
# -------------------------------------------------
print("\n=== SUMMARY OF ALL FEATURES ===")

ttest("LQ80", LQ80[sse], LQ80[mse])
ttest("ND80", ND80[sse], ND80[mse])
ttest("Peak Width", peak_width[sse], peak_width[mse])
ttest("Energy Duration", energy_dur[sse], energy_dur[mse])
ttest("Drift Time (50 percent)", drift_50[sse], drift_50[mse])
ttest("Peak Frequency", peak_freq[sse], peak_freq[mse])
ttest("Spectral Centroid", spec_centroid[sse], spec_centroid[mse])
ttest("AvsE", avse[sse], avse[mse])
ttest("HFER", hfer_vals[sse], hfer_vals[mse])
ttest("Spectral Centroid Power", spec_centroid_power[sse], spec_centroid_power[mse])
ttest("Tail Charge Difference", tail_charge_diff[sse], tail_charge_diff[mse])
ttest("Time to Peak", time_to_peak[sse], time_to_peak[mse])
ttest("Total Power", total_power[sse], total_power[mse])
ttest("Current Skewness", current_skew[sse], current_skew[mse])
ttest("Current Kurtosis", current_kurt[sse], current_kurt[mse])

# -------------------------------------------------
# Save all histograms
# -------------------------------------------------
plot_hist_LQ80(LQ80[sse], LQ80[mse], f"{OUT_DIR}/LQ80_hist.png")
plot_hist_ND80(ND80[sse], ND80[mse], f"{OUT_DIR}/ND80_hist.png")
plot_hist_peak_width(peak_width[sse], peak_width[mse], f"{OUT_DIR}/peak_width_hist.png")
plot_hist_energy_duration(energy_dur[sse], energy_dur[mse], f"{OUT_DIR}/energy_duration_hist.png")
plot_hist_drift_times(drift_50[sse], drift_50[mse], f"{OUT_DIR}/drift_times_hist.png")
plot_hist_peak_frequency(peak_freq[sse], peak_freq[mse], f"{OUT_DIR}/peak_freq_hist.png")
plot_hist_spectral_centroid(spec_centroid[sse], spec_centroid[mse], f"{OUT_DIR}/spectral_centroid_hist.png")
plot_hist_avse(avse[sse], avse[mse], f"{OUT_DIR}/avse_hist.png")
plot_hist_hfer(hfer_vals[sse], hfer_vals[mse], f"{OUT_DIR}/hfer_hist.png")
plot_hist_spectral_centroid_power(spec_centroid_power[sse], spec_centroid_power[mse], f"{OUT_DIR}/spectral_centroid_power_hist.png")
plot_hist_tail_charge_diff(tail_charge_diff[sse], tail_charge_diff[mse], f"{OUT_DIR}/tail_charge_diff_hist.png")
plot_hist_time_to_peak(time_to_peak[sse], time_to_peak[mse], f"{OUT_DIR}/time_to_peak_hist.png")
plot_hist_total_power(total_power[sse], total_power[mse], f"{OUT_DIR}/total_power_hist.png")
plot_hist_current_kurtosis(current_kurt[sse], current_kurt[mse], f"{OUT_DIR}/current_kurtosis_hist.png")
plot_hist_current_skewness(current_skew[sse], current_skew[mse], f"{OUT_DIR}/current_skewness_hist.png")

print("\nSaved ALL plots to graphs/ folder.")
print("Done.\n")