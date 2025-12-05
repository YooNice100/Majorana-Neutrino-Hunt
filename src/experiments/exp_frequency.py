import numpy as np
import os
from scipy.stats import ttest_ind

from src.utils.io import load_hdf5
from src.utils.plots import (
    plot_hist_peak_frequency,
    plot_hist_spectral_centroid,
)
from src.parameters.frequency_domain import (
    compute_peak_frequency,
    compute_spectral_centroid,
)
from src.utils.stats import make_sse_mse_masks


# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------
DATA_PATH = "data/MJD_Train_2.hdf5"
OUT_DIR = "graphs"
os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
print(f"Loading: {DATA_PATH}")
data = load_hdf5(DATA_PATH)
N = data["raw_waveform"].shape[0]
print(f"Events loaded: {N}")


# ------------------------------------------------------------
# Build SSE / MSE masks
# ------------------------------------------------------------
sse, mse = make_sse_mse_masks(data)


# ------------------------------------------------------------
# Compute frequency features
# ------------------------------------------------------------
waveforms = data["raw_waveform"]

peak_freq = []
centroid = []

for wf in waveforms:
    peak_freq.append(compute_peak_frequency(wf))
    centroid.append(compute_spectral_centroid(wf))

peak_freq = np.array(peak_freq)
centroid = np.array(centroid)


# ------------------------------------------------------------
# Split into SSE/MSE groups
# ------------------------------------------------------------
pf_sse = peak_freq[sse]
pf_mse = peak_freq[mse]

sc_sse = centroid[sse]
sc_mse = centroid[mse]


# ------------------------------------------------------------
# T-test function
# ------------------------------------------------------------
def ttest(x, y):
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    t, p = ttest_ind(x, y, equal_var=False)
    return t, p


print("\n=== Frequency Domain Features ===")

t, p = ttest(pf_sse, pf_mse)
print(f"Peak Frequency: t = {t:.2f}, p = {p:.3e}")

t, p = ttest(sc_sse, sc_mse)
print(f"Spectral Centroid: t = {t:.2f}, p = {p:.3e}")


# ------------------------------------------------------------
# Save plots
# ------------------------------------------------------------
plot_hist_peak_frequency(pf_sse, pf_mse, save_path=f"{OUT_DIR}/peak_frequency_hist.png")
plot_hist_spectral_centroid(sc_sse, sc_mse, save_path=f"{OUT_DIR}/spectral_centroid_hist.png")

print("\nSaved frequency-domain plots to graphs/.\n")
