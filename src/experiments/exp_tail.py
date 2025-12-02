# src/experiments/exp_tail.py

import os
import numpy as np

from ..utils.io import load_hdf5
from ..utils.stats import make_strict_masks, summarize_feature
from ..parameters.tail_features import compute_ND80
from ..utils.plots import plot_hist_ND80


# Path to the training file (has labels)
DATA_PATH = "data/MJD_Train_2.hdf5"

# Where to save figures
OUT_DIR = "graphs"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load HDF5 file
    print("Loading file:", DATA_PATH)
    data = load_hdf5(DATA_PATH)

    # 2. Build strict SSE / MSE masks
    strict_sse, strict_mse = make_strict_masks(data)

    # 3. Compute ND80 (normalized) for every event
    waveforms = data["raw_waveform"]
    n_events = waveforms.shape[0]

    print("Number of events:", n_events)

    nd80_norm_all = np.full(n_events, np.nan, dtype=float)

    for i in range(n_events):
        _, _, nd80_norm = compute_ND80(waveforms[i])
        nd80_norm_all[i] = nd80_norm

    # 4. Split into SSE vs MSE distributions
    nd80_sse = nd80_norm_all[strict_sse]
    nd80_mse = nd80_norm_all[strict_mse]

    # 5. Print basic stats + Welch t-test
    summarize_feature("ND80 (normalized)", nd80_sse, nd80_mse)

    # 6. Save histogram
    plot_path = os.path.join(OUT_DIR, "ND80_hist.png")
    plot_hist_ND80(nd80_sse, nd80_mse, save_path=plot_path)
    print("\nSaved ND80 histogram to:", plot_path)


if __name__ == "__main__":
    main()
