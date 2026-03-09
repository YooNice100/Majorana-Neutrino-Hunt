import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

print("\nStarting classification pipeline...")

# load the data
print("Loading datasets...")

df_train = pd.read_csv("src/data/combined_train_with_labels.csv.gz")
df_test = pd.read_csv("src/data/combined_test_with_labels.csv.gz")

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

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

X_test = X_test[X_train.columns]

print("Number of features:", len(X_train.columns))

# Labels
y_train_low_avse = df_train["psd_label_low_avse"]
y_train_high_avse = df_train["psd_label_high_avse"]
y_train_dcr = df_train["psd_label_dcr"]
y_train_lq = df_train["psd_label_lq"]

y_test_low_avse = df_test["psd_label_low_avse"]
y_test_high_avse = df_test["psd_label_high_avse"]
y_test_dcr = df_test["psd_label_dcr"]
y_test_lq = df_test["psd_label_lq"]

# ------------------------
# low avse
# ------------------------

print("\nTraining LOW AVSE RandomForest...")

rf_low_avse = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)

rf_low_avse.fit(X_train, y_train_low_avse)

pred_low_avse = rf_low_avse.predict(X_test).astype(int)

print("LOW AVSE training complete")

# ------------------------
# high avse
# ------------------------

print("\nTraining HIGH AVSE XGBoost...")

high_avse_threshold = 0.7

high_avse_xgb = XGBClassifier(
    subsample=1.0,
    scale_pos_weight=0.5,
    n_estimators=500,
    max_depth=6,
    learning_rate=0.3,
    gamma=0.1,
    colsample_bytree=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

high_avse_xgb.fit(X_train, y_train_high_avse)

probs_high_avse = high_avse_xgb.predict_proba(X_test)[:, 1]
pred_high_avse = (probs_high_avse >= high_avse_threshold).astype(int)

print("HIGH AVSE training complete")

# ------------------------
# lq
# ------------------------

print("\nTraining LQ RandomForest...")

best_threshold_lq = 0.42

rf_pipe_lq = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    ))
])

rf_pipe_lq.fit(X_train, y_train_lq)

probs_lq = rf_pipe_lq.predict_proba(X_test)[:, 1]
pred_lq = (probs_lq >= best_threshold_lq).astype(int)

print("LQ training complete")

# ------------------------
# dcr
# ------------------------

print("\nTraining DCR Neural Network...")

dcr_nn_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        batch_size=1024,
        max_iter=20,
        alpha=1e-4,
        learning_rate="adaptive",
        random_state=42
    ))
])

dcr_nn_pipe.fit(X_train, y_train_dcr)

probs_dcr = dcr_nn_pipe.predict_proba(X_test)[:, 1]
pred_dcr = (probs_dcr >= 0.5).astype(int)

print("DCR training complete")

# ------------------------
# save predictions
# ------------------------

print("\nSaving prediction file...")

pred_df = pd.DataFrame({
    "id": df_test["id"],
    "pred_low_avse": pred_low_avse,
    "pred_high_avse": pred_high_avse,
    "pred_dcr": pred_dcr,
    "pred_lq": pred_lq
})

pred_df.to_csv("src/results/combined_classification_predictions.csv", index=False)

print("Prediction file saved.")

# ------------------------
# metrics
# ------------------------

metrics = []

def add_metrics(name, y_true, y_pred):
    metrics.append({
        "label": name,
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    })

add_metrics("low_avse", y_test_low_avse, pred_low_avse)
add_metrics("high_avse", y_test_high_avse, pred_high_avse)
add_metrics("dcr", y_test_dcr, pred_dcr)
add_metrics("lq", y_test_lq, pred_lq)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("src/results/classification_metrics.csv", index=False)

print("\nClassification metrics:")
print(metrics_df)

print("\nClassification pipeline complete.")