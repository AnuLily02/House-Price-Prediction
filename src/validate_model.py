from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Locate project files
ROOT = Path(__file__).resolve().parent.parent
train_path = ROOT / "data" / "Cleaned train.csv"

df = pd.read_csv(train_path)

target = "Saleprice"
X = df.drop(columns=[target, "Id"], errors="ignore")
y = df[target]

# Five-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # Fit preprocessing only on the training fold.
    model = Pipeline([
        ("remove_constant_features", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])

    # Train Ridge Regression on the log-transformed target.
    model.fit(X_train, np.log1p(y_train))

    # Convert predictions back to the original target scale.
    predictions = np.expm1(model.predict(X_val))
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)

    results.append({
        "Fold": fold,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    print(
        f"Fold {fold}: "
        f"MAE={mae:,.2f}, "
        f"RMSE={rmse:,.2f}, "
        f"R2={r2:.4f}"
    )

results_df = pd.DataFrame(results)

print("\nCROSS-VALIDATION SUMMARY")
print("-" * 45)

for metric in ["MAE", "RMSE", "R2"]:
    print(
        f"{metric}: "
        f"Mean={results_df[metric].mean():,.4f}, "
        f"Std={results_df[metric].std():,.4f}"
    )

output_path = ROOT / "outputs" / "cross_validation_results.csv"
results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")