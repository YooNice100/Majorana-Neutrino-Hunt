import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("\nGenerating NPML plots...")

os.makedirs("graphs", exist_ok=True)

# -------------------------
# Load predictions
# -------------------------

pred_df = pd.read_csv("results/npml_predictions.csv")

print("Loaded NPML predictions:", pred_df.shape)

bins = np.linspace(0, 4000, 400)

# -------------------------
# LightGBM - All Events
# -------------------------

plt.figure(figsize=(8,4))

plt.hist(
    pred_df["pred_energy_lgb"],
    bins=bins,
    histtype="step",
    log=True,
    label="Predicted Energy (All Events)"
)

plt.xlabel("Energy (keV)")
plt.ylabel("Counts (log scale)")
plt.title("NPML Energy Spectrum (LightGBM)")
plt.legend()
plt.tight_layout()

plt.savefig("graphs/npml_lgb_all.png")
plt.close()

print("Saved graphs/npml_lgb_all.png")

# -------------------------
# LightGBM - PSD Cut
# -------------------------

lgb_cut = pred_df[pred_df["SSE"] == 1]["pred_energy_lgb"]

plt.figure(figsize=(8,4))

plt.hist(
    lgb_cut,
    bins=bins,
    histtype="step",
    log=True,
    label="Predicted Energy (SSE Events)"
)

plt.xlabel("Energy (keV)")
plt.ylabel("Counts (log scale)")
plt.title("NPML Energy Spectrum (LightGBM - PSD Cut)")
plt.legend()
plt.tight_layout()

plt.savefig("graphs/npml_lgb_psd_cut.png")
plt.close()

print("Saved graphs/npml_lgb_psd_cut.png")

# -------------------------
# XGBoost - All Events
# -------------------------

plt.figure(figsize=(8,4))

plt.hist(
    pred_df["pred_energy_xgb"],
    bins=bins,
    histtype="step",
    log=True,
    label="Predicted Energy (All Events)"
)

plt.xlabel("Energy (keV)")
plt.ylabel("Counts (log scale)")
plt.title("NPML Energy Spectrum (XGBoost)")
plt.legend()
plt.tight_layout()

plt.savefig("graphs/npml_xgb_all.png")
plt.close()

print("Saved graphs/npml_xgb_all.png")

# -------------------------
# XGBoost - PSD Cut
# -------------------------

xgb_cut = pred_df[pred_df["SSE"] == 1]["pred_energy_xgb"]

plt.figure(figsize=(8,4))

plt.hist(
    xgb_cut,
    bins=bins,
    histtype="step",
    log=True,
    label="Predicted Energy (SSE Events)"
)

plt.xlabel("Energy (keV)")
plt.ylabel("Counts (log scale)")
plt.title("NPML Energy Spectrum (XGBoost - PSD Cut)")
plt.legend()
plt.tight_layout()

plt.savefig("graphs/npml_xgb_psd_cut.png")
plt.close()

print("Saved graphs/npml_xgb_psd_cut.png")

print("\nNPML plots complete.")