import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

print("\nStarting regression pipeline...")

# load dataset

df_train = pd.read_csv("data/combined_train_with_labels.csv.gz")
df_test = pd.read_csv("data/combined_test_with_labels.csv.gz")

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

# ------------------------
# Load classification predictions
# ------------------------

pred_df = pd.read_csv("results/combined_classification_predictions.csv")

# ------------------------
# Prepare features
# ------------------------

drop_cols = [
    "id",
    "energy_label",
    "psd_label_low_avse",
    "psd_label_high_avse",
    "psd_label_dcr",
    "psd_label_lq"
]

X_train = df_train.drop(columns=drop_cols)
X_test = df_test.drop(columns=drop_cols)

# Ensure column order matches
X_test = X_test[X_train.columns]

y_train_energy = df_train["energy_label"]
y_test_energy = df_test["energy_label"]

print("Number of features:", len(X_train.columns))

# ------------------------
# Train XGBoost Regression
# ------------------------

print("\nTraining XGBoost Regressor...")

xgb_regressor_model = XGBRegressor(
    subsample=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    n_estimators=800,
    max_depth=10,
    learning_rate=0.1,
    colsample_bytree=0.7,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

xgb_regressor_model.fit(X_train, y_train_energy)

# ------------------------
# Predict
# ------------------------

y_pred_xgb = xgb_regressor_model.predict(X_test)

print("Prediction complete")

# ------------------------
# Compute metrics
# ------------------------

mse = mean_squared_error(y_test_energy, y_pred_xgb)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_energy, y_pred_xgb)
r2 = r2_score(y_test_energy, y_pred_xgb)

print("\nRegression metrics:")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)

# ------------------------
# Save predictions
# ------------------------

predictions = pd.DataFrame({
    "id": df_test["id"],
    "true_energy": y_test_energy,
    "pred_energy": y_pred_xgb
})

predictions.to_csv("results/regression_predictions.csv", index=False)

# ------------------------
# Save metrics
# ------------------------

metrics_df = pd.DataFrame([{
    "model": "xgboost",
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2": r2
}])

metrics_df.to_csv("results/regression_metrics.csv", index=False)

print("\nRegression pipeline complete.")