from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. PATHS AND DATA
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

train = pd.read_csv(DATA_DIR / "Cleaned train.csv")
test = pd.read_csv(DATA_DIR / "Cleaned test.csv")

TARGET = "Saleprice"

X = train.drop(columns=[TARGET]).copy()
y = train[TARGET].copy()

# Confirm the training and test feature columns match.
if list(X.columns) != list(test.columns):
    raise ValueError("Training and test feature columns do not match.")

# Remove features with no variation in the training data.
constant_features = X.columns[X.nunique() <= 1].tolist()
X = X.drop(columns=constant_features)
test = test.drop(columns=constant_features)

# Exclude the identifier from the model.
if "Id" in X.columns:
    X = X.drop(columns=["Id"])
    test = test.drop(columns=["Id"])

# Check numeric values before training.
if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Training features contain NaN or infinite values.")

if not np.isfinite(test.to_numpy(dtype=float)).all():
    raise ValueError("Test features contain NaN or infinite values.")

if not np.isfinite(y.to_numpy(dtype=float)).all():
    raise ValueError("Target contains NaN or infinite values.")

print("Training rows:", len(X))
print("Features after removing constants and Id:", X.shape[1])
print("Removed constant features:", constant_features)


# --------------------------------------------------
# 2. TRAIN / VALIDATION SPLIT
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# --------------------------------------------------
# 3. MODELS
# --------------------------------------------------

models = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression(),
    ),
    "Ridge Regression": make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    ),
}


# --------------------------------------------------
# 4. TRAIN AND EVALUATE
# --------------------------------------------------

results = []
trained_models = {}

target_versions = {
    "raw": False,
    "log1p": True,
}

for target_name, use_log in target_versions.items():
    if use_log:
        if (y_train < 0).any():
            raise ValueError("Cannot log-transform negative target values.")
        y_fit = np.log1p(y_train)
    else:
        y_fit = y_train

    for model_name, model in models.items():
        print(f"\nTraining {model_name} with {target_name} target...")

        model.fit(X_train, y_fit)

        predictions = model.predict(X_val)

        # Convert log predictions back to the original scale.
        if use_log:
            predictions = np.expm1(predictions)

        mae = mean_absolute_error(y_val, predictions)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        r2 = r2_score(y_val, predictions)

        results.append({
            "model": model_name,
            "target_version": target_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        trained_models[(model_name, target_name)] = model

        print(f"MAE : {mae:,.2f}")
        print(f"RMSE: {rmse:,.2f}")
        print(f"R2  : {r2:.4f}")


# --------------------------------------------------
# 5. SAVE MODEL COMPARISON
# --------------------------------------------------

comparison = pd.DataFrame(results).sort_values("RMSE")

comparison_path = OUTPUT_DIR / "model_comparison.csv"
comparison.to_csv(comparison_path, index=False)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

# Select the model with the lowest validation RMSE.
best_row = comparison.iloc[0]
best_key = (best_row["model"], best_row["target_version"])
best_model = trained_models[best_key]
use_log = best_key[1] == "log1p"

print("\nBest validation model:", best_key[0])
print("Target transformation:", best_key[1])
print(f"Validation RMSE: {best_row['RMSE']:,.2f}")

# Save model together with metadata needed for predictions.
model_bundle = {
    "model": best_model,
    "feature_columns": list(X.columns),
    "target_transformation": best_key[1],
    "target_column": TARGET,
    "excluded_features": ["Id"] + constant_features,
}

model_path = MODEL_DIR / "house_price_model.joblib"
joblib.dump(model_bundle, model_path)

# --------------------------------------------------
# 6. GENERATE TEST PREDICTIONS
# --------------------------------------------------

test_predictions = best_model.predict(test)

if use_log:
    test_predictions = np.expm1(test_predictions)

# Keep predictions non-negative.
test_predictions = np.maximum(test_predictions, 0)

prediction_output = pd.DataFrame({
    "Id": pd.read_csv(DATA_DIR / "Cleaned test.csv")["Id"],
    "Predicted_Saleprice": test_predictions,
})

prediction_path = OUTPUT_DIR / "predictions.csv"
prediction_output.to_csv(prediction_path, index=False)

print("\nSaved model:", model_path)
print("Saved comparison:", comparison_path)
print("Saved test predictions:", prediction_path)
print("\nSample predictions:")
print(prediction_output.head(10).to_string(index=False))