import numpy as np
import os
from scipy.stats import ttest_ind

from src.utils.io import load_hdf5
from src.utils.plots import (
    plot_hist_peak_width,
    plot_hist_energy_duration,
    plot_hist_drift_times,
)
from src.parameters.time_domain import (
    compute_peak_width_25_75,
    compute_energy_duration,
    estimate_tp0_threshold,
    compute_drift_times,
)


DATA_PATH = "data/MJD_Train_2.hdf5"
OUT_DIR = "graphs"
os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
print(f"Loading file: {DATA_PATH}")
data = load_hdf5(DATA_PATH)
N = data["raw_waveform"].shape[0]
print(f"Number of events: {N}")


# ------------------------------------------------------------
# Strict SSE / MSE labels
# ------------------------------------------------------------
low  = data["psd_label_low_avse"].astype(bool)
high = data["psd_label_high_avse"].astype(bool)
dcr  = data["psd_label_dcr"].astype(bool)
lq   = data["psd_label_lq"].astype(bool)

strict_sse = low & (~high) & dcr & lq
strict_mse = (~low) & high & (~dcr) & (~lq)


# ------------------------------------------------------------
# Compute features
# ------------------------------------------------------------
waveforms = data["raw_waveform"]

peak_width = []
energy_dur = []
drift_50 = []  # we will save the 50 percent drift time


for wf in waveforms:
    # Width 25–75
    width, _, _ = compute_peak_width_25_75(wf)
    peak_width.append(width)

    # Energy Duration 90 percent
    ed = compute_energy_duration(wf)
    energy_dur.append(ed)

    # Drift times (need tp0)
    tp0 = estimate_tp0_threshold(wf)
    if tp0 is None:
        drift_50.append(np.nan)
        continue

    _, t50, _ = compute_drift_times(wf, tp0)
    drift_50.append(t50)


peak_width = np.array(peak_width)
energy_dur = np.array(energy_dur)
drift_50 = np.array(drift_50)


# ------------------------------------------------------------
# Split into SSE / MSE groups
# ------------------------------------------------------------
pw_sse = peak_width[strict_sse]
pw_mse = peak_width[strict_mse]

ed_sse = energy_dur[strict_sse]
ed_mse = energy_dur[strict_mse]

dt_sse = drift_50[strict_sse]
dt_mse = drift_50[strict_mse]


# ------------------------------------------------------------
# T-tests for each feature
# ------------------------------------------------------------
def ttest(x, y):
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    t, p = ttest_ind(x, y, equal_var=False)
    return t, p


print("\n=== Time-Domain Features ===")

# Peak Width
t, p = ttest(pw_sse, pw_mse)
print(f"Peak Width: t = {t:.2f}, p = {p:.3e}")

# Energy Duration
t, p = ttest(ed_sse, ed_mse)
print(f"Energy Duration: t = {t:.2f}, p = {p:.3e}")

# Drift Time 50 percent
t, p = ttest(dt_sse, dt_mse)
print(f"Drift Time (50 percent): t = {t:.2f}, p = {p:.3e}")


# ------------------------------------------------------------
# Save plots
# ------------------------------------------------------------
plot_hist_peak_width(pw_sse, pw_mse, save_path=f"{OUT_DIR}/peak_width_hist.png")
plot_hist_energy_duration(ed_sse, ed_mse, save_path=f"{OUT_DIR}/energy_duration_hist.png")
plot_hist_drift_times(dt_sse, dt_mse, save_path=f"{OUT_DIR}/drift_50_hist.png")

print("\nSaved all time-domain plots to graphs/.\n")
