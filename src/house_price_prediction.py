"""House price prediction.

Version 1 (default):  Linear Regression only
Version 2 (--compare): Linear Regression vs Decision Tree vs Random Forest

Run from the project root:
    python src/house_price_prediction.py
    python src/house_price_prediction.py --compare
    python src/house_price_prediction.py --target SalePrice     # if the target isn't auto-detected
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42


# ---------- Step 1: load data ----------
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if Path(test_path).exists() else None
    return train, test


def find_target(df, target=None):
    if target:
        return target
    lower = {c.lower(): c for c in df.columns}   # case-insensitive match
    for c in ["saleprice", "price", "house_price"]:
        if c in lower:
            return lower[c]
    print(f"[info] target not recognised, using last column: {df.columns[-1]}")
    return df.columns[-1]


# ---------- Step 2: preprocessing ----------
def build_preprocessor(X):
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns
    numeric = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])


# ---------- Step 3: models ----------
def get_models(compare):
    models = {"LinearRegression": LinearRegression()}
    if compare:
        models["DecisionTree"] = DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE)
        models["RandomForest"] = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    return models


def make_pipeline(X, regressor):
    return Pipeline([("prep", build_preprocessor(X)), ("model", regressor)])


# ---------- Step 4: evaluation ----------
def evaluate(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "data" / "train.csv"))
    ap.add_argument("--test", default=str(ROOT / "data" / "test.csv"))
    ap.add_argument("--target", default=None)
    ap.add_argument("--compare", action="store_true", help="also train Decision Tree and Random Forest")
    args = ap.parse_args()

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)

    train, test = load_data(args.train, args.test)
    target = find_target(train, args.target)
    train = train.dropna(subset=[target])
    id_col = next((c for c in ["Id", "ID", "id"] if c in train.columns), None)

    y = train[target]
    X = train.drop(columns=[c for c in [target, id_col] if c])
    print(f"[info] rows={len(X)}  features={X.shape[1]}  target={target}")

    # Hold out 20% of train to measure how the model does on unseen houses
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # ---------- Step 5: train + compare ----------
    results = []
    for name, reg in get_models(args.compare).items():
        pipe = make_pipeline(X, reg)
        pipe.fit(X_tr, y_tr)
        train_r2 = r2_score(y_tr, pipe.predict(X_tr))
        m = evaluate(y_val, pipe.predict(X_val))
        results.append({"model": name, "train_R2": train_r2, **m})
        print(f"{name:17s} train_R2={train_r2:.3f} | val RMSE={m['RMSE']:.2f} MAE={m['MAE']:.2f} R2={m['R2']:.3f}")

    comparison = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    comparison.to_csv(ROOT / "outputs" / "model_comparison.csv", index=False)
    best_name = comparison.loc[0, "model"]
    print(f"[best] {best_name}")

    # ---------- Step 6: retrain best on ALL train data, save ----------
    best = make_pipeline(X, get_models(args.compare)[best_name])
    best.fit(X, y)
    joblib.dump(best, ROOT / "models" / "house_price_model.pkl")

    # ---------- Step 7: predict on the test set ----------
    if test is not None:
        ids = test[id_col] if id_col and id_col in test.columns else pd.Series(range(len(test)))
        X_test = test.drop(columns=[c for c in [target, id_col] if c and c in test.columns])
        X_test = X_test.reindex(columns=X.columns)
        preds = best.predict(X_test)
        out = pd.DataFrame({"Id": ids.values, f"Predicted_{target}": preds})
        if target in test.columns:
            out[f"Actual_{target}"] = test[target].values
            print("[test]", evaluate(test[target], preds))
    else:
        preds = best.predict(X_val)
        out = pd.DataFrame({"Actual": y_val.values, "Predicted": preds})
    out.to_csv(ROOT / "outputs" / "predictions.csv", index=False)
    print("[done] model, predictions and comparison saved")


if __name__ == "__main__":
    main()
