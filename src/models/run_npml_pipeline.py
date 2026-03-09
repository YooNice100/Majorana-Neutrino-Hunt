import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier, XGBRegressor


print("\nStarting NPML pipeline...")

# -------------------------
# Load NPML feature files
# -------------------------

eunice = pd.read_csv("feature_inputs/eunice_combined_npml.csv.gz")
nomin = pd.read_csv("feature_inputs/nomin_combined_npml_n.csv.gz")
prithvi = pd.read_csv("feature_inputs/prithvi_combined_npml.csv.gz")
jade = pd.read_csv("feature_inputs/jade_npml_features.csv")

print("Feature shapes:")
print(eunice.shape, nomin.shape, prithvi.shape, jade.shape)

# -------------------------
# Merge NPML features
# -------------------------

df = eunice.merge(nomin, on="id")
df = df.merge(prithvi, on="id")
df = df.merge(jade, on="id")

print("Merged NPML shape:", df.shape)

# -------------------------
# Load training data
# -------------------------

train_df = pd.read_csv("data/combined_train_with_labels.csv.gz")

classification_label_cols = [
    "psd_label_lq",
    "psd_label_high_avse",
    "psd_label_low_avse",
    "psd_label_dcr"
]

non_label_cols = ["id", "energy_label"]

feature_cols = [
    col for col in train_df.columns
    if col not in classification_label_cols + non_label_cols
]

print("Number of features:", len(feature_cols))

X_train = train_df[feature_cols]

y_train_high_avse = train_df["psd_label_high_avse"]
y_train_low_avse = train_df["psd_label_low_avse"]
y_train_dcr = train_df["psd_label_dcr"]
y_train_lq = train_df["psd_label_lq"]
y_train_energy = train_df["energy_label"]

X_npml = df[feature_cols]

# -------------------------
# Classification models
# -------------------------

print("\nTraining classification models...")

high_avse_xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.3,
    gamma=0.1,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

high_avse_xgb.fit(X_train, y_train_high_avse)

rf_low_avse = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf_low_avse.fit(X_train, y_train_low_avse)

rf_pipe_lq = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipe_lq.fit(X_train, y_train_lq)

dcr_nn_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(64,32),
        max_iter=20,
        random_state=42
    ))
])

dcr_nn_pipe.fit(X_train, y_train_dcr)

# -------------------------
# Predict classification
# -------------------------

print("Predicting NPML labels...")

pred_high_avse = (high_avse_xgb.predict_proba(X_npml)[:,1] >= 0.7).astype(int)
pred_lq = (rf_pipe_lq.predict_proba(X_npml)[:,1] >= 0.42).astype(int)
pred_low_avse = rf_low_avse.predict(X_npml).astype(int)
pred_dcr = (dcr_nn_pipe.predict_proba(X_npml)[:,1] >= 0.5).astype(int)

# -------------------------
# Regression models
# -------------------------

print("Training regression models...")

lgb_model = lgb.LGBMRegressor(
    learning_rate=0.03,
    n_estimators=5000,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(X_train, y_train_energy)

xgb_reg = XGBRegressor(
    n_estimators=800,
    max_depth=10,
    learning_rate=0.1,
    colsample_bytree=0.7,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb_reg.fit(X_train, y_train_energy)

print("Predicting NPML energy...")

pred_energy_lgb = lgb_model.predict(X_npml)
pred_energy_xgb = xgb_reg.predict(X_npml)

# -------------------------
# Combine predictions
# -------------------------

pred_df = pd.DataFrame({
    "id": df["id"],
    "pred_lq": pred_lq,
    "pred_high_avse": pred_high_avse,
    "pred_low_avse": pred_low_avse,
    "pred_dcr": pred_dcr,
    "pred_energy_lgb": pred_energy_lgb,
    "pred_energy_xgb": pred_energy_xgb
})

pred_df["SSE"] = (
    (pred_df["pred_lq"]==1) &
    (pred_df["pred_high_avse"]==1) &
    (pred_df["pred_low_avse"]==1) &
    (pred_df["pred_dcr"]==1)
).astype(int)

pred_df.to_csv("results/npml_predictions.csv", index=False)

print("Saved results/npml_predictions.csv")
print("\nNPML pipeline complete.")