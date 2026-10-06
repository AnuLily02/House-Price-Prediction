
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

model_path = ROOT / "models" / "house_price_model.joblib"
test_path = ROOT / "data" / "Cleaned test.csv"
output_path = ROOT / "outputs" / "predictions_reloaded_model.csv"

# Load the saved model and metadata.
bundle = joblib.load(model_path)
model = bundle["model"]
feature_columns = bundle["feature_columns"]
transformation = bundle["target_transformation"]

# Load the test data.
test = pd.read_csv(test_path)
ids = test["Id"].copy() if "Id" in test.columns else None

# Apply the same feature exclusions as during training.
X_test = test.drop(columns=["Id"], errors="ignore")
X_test = X_test.drop(
    columns=bundle["excluded_features"],
    errors="ignore",
)

# Ensure features are in the exact expected order.
missing = set(feature_columns) - set(X_test.columns)
extra = set(X_test.columns) - set(feature_columns)

if missing or extra:
    raise ValueError(
        f"Feature mismatch. Missing: {missing}; Extra: {extra}"
    )

X_test = X_test[feature_columns]

# Predict using the saved model.
predictions = model.predict(X_test)

if transformation == "log1p":
    predictions = np.expm1(predictions)

predictions = np.maximum(predictions, 0)

output = pd.DataFrame({
    "Id": ids if ids is not None else range(len(predictions)),
    "Predicted_Saleprice": predictions,
})

output.to_csv(output_path, index=False)

print("Loaded saved model successfully.")
print("Target transformation:", transformation)
print("Number of predictions:", len(output))
print("All predictions finite:", np.isfinite(predictions).all())
print("Minimum prediction:", predictions.min())
print("Maximum prediction:", predictions.max())
print("Saved predictions to:", output_path)

print("\nFirst 10 predictions:")
print(output.head(10).to_string(index=False))
