# ============================================================
# Eunice Feature Extraction Script
# Generates CSVs for: LQ80, ND80, PPR, Energy Duration, Spectral Centroid
# For Train 2 and Test 2 datasets.
# ============================================================

import os
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from numpy.fft import rfft, rfftfreq

# ------------------------------------------------------------
# Paths (relative to this script)
# ------------------------------------------------------------
TRAIN_PATH = "../../data/old/MJD_Train_2.hdf5"
TEST_PATH  = "../../data/old/MJD_Test_2.hdf5"

TRAIN_OUT = "train_csv"
TEST_OUT  = "test_csv"

os.makedirs(TRAIN_OUT, exist_ok=True)
os.makedirs(TEST_OUT, exist_ok=True)


# ------------------------------------------------------------
# 1. Baseline estimation
# ------------------------------------------------------------
def estimate_baseline(y, n_samples=200):
    y = np.asarray(y, float)
    y0 = y[:n_samples]
    return float(np.mean(y0)), float(np.std(y0))


# ------------------------------------------------------------
# 2. Double exponential model for PZ correction
# ------------------------------------------------------------
def exponential(t, a, tau1, b, tau2):
    return a * np.exp(-t / tau1) + b * np.exp(-t / tau2)


# ------------------------------------------------------------
# 3. Pole Zero Correction
# ------------------------------------------------------------
def pole_zero_correction(waveform, use_pz=False):
    y = np.asarray(waveform, float)
    if not use_pz:
        return y, y

    peak_value = np.max(y)
    t98_idx = np.where(y >= 0.98 * peak_value)[0]
    if len(t98_idx) == 0:
        return y, y
    t98 = int(t98_idx[0])

    tail = y[t98:]
    t = np.arange(len(tail))

    try:
        params, _ = curve_fit(
            exponential,
            t,
            tail,
            p0=[peak_value, 300, peak_value * 0.1, 1500],
            bounds=([0, 10, 0, 10],
                    [peak_value * 2, 5000, peak_value * 2, 5000]),
            maxfev=4000
        )

        model = exponential(t, *params)

        f_t0 = np.mean(tail[:5])
        f_pz = f_t0 / model

        corrected_tail = tail * f_pz

        waveform_pz = y.copy()
        waveform_pz[t98:] = corrected_tail
        return waveform_pz, corrected_tail

    except Exception:
        return y, y


# ------------------------------------------------------------
# 4. Frequency Spectrum
# ------------------------------------------------------------
def compute_frequency_spectrum(waveform, sample_spacing=1.0):
    wf = np.asarray(waveform, float)
    N = len(wf)

    yf = rfft(wf)
    xf = rfftfreq(N, d=sample_spacing)

    amp = np.abs(yf) * 2.0 / N
    return xf, amp


# ------------------------------------------------------------
# 5. Feature: LQ80
# ------------------------------------------------------------
def compute_LQ80(waveform):
    waveform_pz, _ = pole_zero_correction(waveform, use_pz=True)

    y = np.asarray(waveform, float)
    yc = np.asarray(waveform_pz, float)

    baseline, _ = estimate_baseline(y)
    peak = float(np.max(y))
    target = baseline + 0.80 * (peak - baseline)

    idx = np.where(y >= target)[0]
    if len(idx) == 0:
        return np.nan
    i80 = int(idx[0])

    t = np.arange(len(y), dtype=float)

    area_raw = np.trapezoid(y[i80:], t[i80:])
    area_corr = np.trapezoid(yc[i80:], t[i80:])

    return float(area_raw - area_corr)


# ------------------------------------------------------------
# 6. Feature: ND80
# ------------------------------------------------------------
def compute_ND80(waveform, n_pre=200):
    y = np.asarray(waveform, float)

    baseline, _ = estimate_baseline(y, n_samples=n_pre)
    peak_idx = int(np.argmax(y))
    peak_val = float(y[peak_idx])
    amp = peak_val - baseline

    if amp <= 0:
        return np.nan

    level80 = baseline + 0.80 * amp
    above = np.where(y >= level80)[0]
    if len(above) == 0:
        return np.nan

    i80 = int(above[0])
    if i80 >= peak_idx:
        return 0.0

    seg = y[i80:peak_idx + 1]
    depth = level80 - seg
    depth[depth < 0] = 0

    depth_abs = float(np.max(depth))
    return depth_abs / amp


# ------------------------------------------------------------
# 7. Peak Plateau Ratio (PPR)
# ------------------------------------------------------------
def compute_PPR(waveform, n_plateau=300):
    y = np.asarray(waveform, float)
    peak = float(np.max(y))
    if peak <= 0:
        return np.nan
    plateau = float(np.mean(y[-n_plateau:]))
    return plateau / peak


# ------------------------------------------------------------
# 8. Energy Duration (90 percent cumulative energy)
# ------------------------------------------------------------
def compute_energy_duration(waveform, threshold=0.9):
    y = np.asarray(waveform, float)
    energy = y**2
    total = float(np.sum(energy))
    if total == 0:
        return np.nan

    cumulative = np.cumsum(energy)
    target = threshold * total
    idxs = np.where(cumulative >= target)[0]
    if len(idxs) == 0:
        return np.nan
    return int(idxs[0])


# ------------------------------------------------------------
# 9. Spectral Centroid
# ------------------------------------------------------------
def compute_spectral_centroid(waveform, sample_spacing=1.0):
    freqs, amp = compute_frequency_spectrum(waveform, sample_spacing)
    total_amp = np.sum(amp)
    if total_amp == 0:
        return 0.0
    centroid = np.sum(freqs * amp) / total_amp
    return float(centroid)


# ------------------------------------------------------------
# RUN EXTRACTION
# ------------------------------------------------------------
def extract_features(path, suffix, out_dir):
    print(f"\nLoading {path}")
    with h5py.File(path, "r") as f:
        waveforms = np.array(f["raw_waveform"])
        ids = np.array(f["id"])

    print(f"Total waveforms: {len(waveforms)}")

    # ---------- COMPUTE FEATURES ----------
    LQ80_vals = []
    ND80_vals = []
    PPR_vals = []
    ED_vals = []
    SC_vals = []

    for i, wf in enumerate(waveforms):
        if i % 5000 == 0:
            print(f"  Processing {i}/{len(waveforms)}")

        LQ80_vals.append(compute_LQ80(wf))
        ND80_vals.append(compute_ND80(wf))
        PPR_vals.append(compute_PPR(wf))
        ED_vals.append(compute_energy_duration(wf))
        SC_vals.append(compute_spectral_centroid(wf))

    # ---------- SAVE CSVs ----------
    formatted_ids = [f"{id_}_{suffix}" for id_ in ids]

    def save_feature(values, name):
        df = pd.DataFrame({"id": formatted_ids, name: values})
        out_path = os.path.join(out_dir, f"{name}_{suffix}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {name} → {out_path}")

    save_feature(LQ80_vals, "LQ80")
    save_feature(ND80_vals, "ND80")
    save_feature(PPR_vals, "PPR")
    save_feature(ED_vals, "EnergyDuration")
    save_feature(SC_vals, "SpectralCentroid")


# ------------------------------------------------------------
# EXECUTE
# ------------------------------------------------------------
if __name__ == "__main__":
    extract_features(TRAIN_PATH, "train2", TRAIN_OUT)
    extract_features(TEST_PATH,  "test2",  TEST_OUT)

    print("\nCompleted ALL feature extraction.\n")
