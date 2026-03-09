import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("\nStarting visualization...")

# load data

df_test = pd.read_csv("data/combined_test_with_labels.csv.gz")
reg_preds = pd.read_csv("results/regression_predictions.csv")
class_preds = pd.read_csv("results/combined_classification_predictions.csv")

print("Loaded datasets")

# -----------------------
# Merge predictions
# -----------------------

df_test_plot = df_test.copy()

# merge regression predictions
df_test_plot = df_test_plot.merge(
    reg_preds[["id", "pred_energy"]],
    on="id",
    how="left"
)

# merge classification predictions
df_test_plot = df_test_plot.merge(
    class_preds,
    on="id",
    how="left"
)

print("Merged dataframe shape:", df_test_plot.shape)

# -----------------------
# Energy spectrum (all events)
# -----------------------

bins = np.linspace(0, 4000, 400)

plt.figure(figsize=(7,4))

plt.hist(
    df_test_plot["energy_label"],
    bins=bins,
    histtype="step",
    log=True,
    label="True Energy"
)

plt.hist(
    df_test_plot["pred_energy"],
    bins=bins,
    histtype="step",
    log=True,
    label="Predicted Energy (XGB)"
)

plt.xlabel("Energy (keV)")
plt.ylabel("Counts")
plt.title("Energy Spectrum: All Test Events")
plt.legend()
plt.tight_layout()

plt.savefig("graphs/energy_spectrum_all_events.png")
plt.close()

print("Saved energy_spectrum_all_events.png")

# -----------------------
# PSD cuts
# -----------------------

mask_true_pass = (
    (df_test_plot["psd_label_low_avse"] == 1) &
    (df_test_plot["psd_label_high_avse"] == 1) &
    (df_test_plot["psd_label_dcr"] == 1) &
    (df_test_plot["psd_label_lq"] == 1)
)

mask_pred_pass = (
    (df_test_plot["pred_low_avse"] == 1) &
    (df_test_plot["pred_high_avse"] == 1) &
    (df_test_plot["pred_dcr"] == 1) &
    (df_test_plot["pred_lq"] == 1)
)

print("True pass fraction:", mask_true_pass.mean())
print("Pred pass fraction:", mask_pred_pass.mean())

# -----------------------
# Energy spectrum after PSD cuts
# -----------------------

plt.figure(figsize=(7,4))

plt.hist(
    df_test_plot.loc[mask_true_pass, "energy_label"],
    bins=bins,
    histtype="step",
    log=True,
    label="True Energy (True PSD Pass)"
)

plt.hist(
    df_test_plot.loc[mask_pred_pass, "pred_energy"],
    bins=bins,
    histtype="step",
    log=True,
    label="Predicted Energy (Predicted PSD Pass)"
)

plt.xlabel("Energy (keV)")
plt.ylabel("Counts")
plt.title("Energy Spectrum After PSD Cuts (True vs Predicted)")
plt.legend()
plt.tight_layout()

plt.savefig("graphs/energy_spectrum_after_psd_cut.png")
plt.close()

print("Saved energy_spectrum_after_psd_cut.png")

print("\nVisualization complete.")