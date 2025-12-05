import os
import numpy as np

# Load helpers
from src.utils.io import load_hdf5, sample_random_event
from src.utils.plots import (
    plot_LQ80_waveform,
    plot_ND80_waveform,
    plot_peak_width_waveform,
    plot_drift_times_waveform,
    plot_fft_linear,
    plot_fft_log,
)

# ------------------------------------------------------------
# 1. Path to your dataset
# ------------------------------------------------------------
DATA_PATH = "data/MJD_Train_2.hdf5"

# ------------------------------------------------------------
# 2. Output directory for graphs
# ------------------------------------------------------------
OUT_DIR = "graphs"

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 3. Load dataset + sample random event
# ------------------------------------------------------------
print("Loading data...")
data = load_hdf5(DATA_PATH)
event = sample_random_event(data)

waveform = event["waveform"]
tp0 = int(event["tp0"])

print(f"Random event index: {event['index']}")
print("Waveform length:", len(waveform))


# ------------------------------------------------------------
# 4. Generate plots (each saves automatically)
# ------------------------------------------------------------
print("Saving plots into 'graphs/'...")

plot_LQ80_waveform(waveform, save_path=f"{OUT_DIR}/LQ80.png")
plot_ND80_waveform(waveform, save_path=f"{OUT_DIR}/ND80.png")
plot_peak_width_waveform(waveform, save_path=f"{OUT_DIR}/peak_width.png")
plot_drift_times_waveform(waveform, tp0, save_path=f"{OUT_DIR}/drift_times.png")
plot_fft_linear(waveform, save_path=f"{OUT_DIR}/fft_linear.png")
plot_fft_log(waveform, save_path=f"{OUT_DIR}/fft_log.png")

print("Done! Check the graphs folder.")