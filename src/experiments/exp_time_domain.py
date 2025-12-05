import numpy as np
import os
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from src.utils.stats import make_sse_mse_masks
from src.utils.io import load_hdf5
from src.utils.plots import (
    plot_hist_peak_width,
    plot_hist_energy_duration,
    plot_hist_drift_times,
    plot_hist_avse,
)
from src.parameters.time_domain import (
    compute_peak_width_25_75,
    compute_energy_duration,
    estimate_tp0_threshold,
    compute_drift_times,
    compute_avse,
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
# SSE / MSE labels
# ------------------------------------------------------------
sse, mse = make_sse_mse_masks(data)

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
pw_sse = peak_width[sse]
pw_mse = peak_width[mse]

ed_sse = energy_dur[sse]
ed_mse = energy_dur[mse]

dt_sse = drift_50[sse]
dt_mse = drift_50[mse]


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

# --- AvsE ---
def run_avse_experiment(data_path="data/MJD_Train_2.hdf5", out_dir="graphs"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading: {data_path}")
    d = load_hdf5(data_path)

    W = d["raw_waveform"]
    E = d["energy_label"]

    # compute AvsE using your implementation in parameters/time_domain.py
    avse, _ = compute_avse(W, E)

    # make sse / mse masks
    sse_mask, mse_mask = make_sse_mse_masks(d)

    # split
    sse_vals = avse[sse_mask]
    mse_vals = avse[mse_mask]

    # plot
    save_path = os.path.join(out_dir, "avse_hist.png")
    plot_hist_avse(sse_vals, mse_vals, save_path=save_path)
    print(f"Saved: {save_path}")

    # quick stats (Mann–Whitney + AUC with SSE=1, MSE=0)
    sse_vals = sse_vals[np.isfinite(sse_vals)]
    mse_vals = mse_vals[np.isfinite(mse_vals)]
    if len(sse_vals) and len(mse_vals):
        U, p = mannwhitneyu(sse_vals, mse_vals, alternative="two-sided")
        y = np.concatenate([np.ones_like(sse_vals), np.zeros_like(mse_vals)])
        f = np.concatenate([sse_vals, mse_vals])
        try:
            auc = roc_auc_score(y, f)
        except Exception:
            auc = np.nan
        print(f"AvsE: Mann–Whitney U={U:.3g}, p={p:.3e}, AUC={auc if np.isfinite(auc) else np.nan:.3f}")
    else:
        print("AvsE: insufficient finite values for statistics.")

