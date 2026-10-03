import pandas as pd
import numpy as np

# ==============================
# LOAD DATA
# ==============================

train_path = "../data/Cleaned train.csv"
test_path = "../data/Cleaned test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target = "Saleprice"

print("=" * 60)
print("HOUSE PRICE PREDICTION - TARGET INSPECTION")
print("=" * 60)

# ==============================
# 1. TARGET INFORMATION
# ==============================

y = train[target]

print("\n1. TARGET INFORMATION")
print("-" * 60)

print("Target column:", target)
print("Target data type:", y.dtype)
print("Number of values:", len(y))

print("\nTarget statistics:")
print("Minimum :", y.min())
print("Maximum :", y.max())
print("Mean    :", y.mean())
print("Median  :", y.median())
print("Std Dev :", y.std())

print("\nTarget quantiles:")
print(y.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]))

# ==============================
# 2. SAMPLE TARGET VALUES
# ==============================

print("\n2. SAMPLE TARGET VALUES")
print("-" * 60)

print(y.head(10).to_string(index=False))

# ==============================
# 3. TARGET MISSING VALUES
# ==============================

print("\n3. TARGET MISSING VALUES")
print("-" * 60)

print("Missing target values:", y.isnull().sum())

# ==============================
# 4. TARGET SKEWNESS
# ==============================

print("\n4. TARGET SKEWNESS")
print("-" * 60)

print("Saleprice skewness:", y.skew())

if y.min() >= 0:
    log_y = np.log1p(y)
    print("log1p(Saleprice) skewness:", log_y.skew())

# ==============================
# 5. TRAIN / TEST COLUMNS
# ==============================

print("\n5. TRAIN / TEST COLUMN CHECK")
print("-" * 60)

train_features = train.drop(columns=[target])

print("Train feature count:", train_features.shape[1])
print("Test feature count :", test.shape[1])

print("Same columns:", list(train_features.columns) == list(test.columns))

missing_in_test = set(train_features.columns) - set(test.columns)
extra_in_test = set(test.columns) - set(train_features.columns)

print("Columns missing from test:", missing_in_test)
print("Extra columns in test:", extra_in_test)

# ==============================
# 6. ZERO-VARIANCE FEATURES
# ==============================

print("\n6. ZERO-VARIANCE FEATURES")
print("-" * 60)

zero_variance = train_features.columns[
    train_features.nunique() <= 1
].tolist()

print("Number of zero-variance features:", len(zero_variance))

if zero_variance:
    print("Zero-variance columns:")
    print(zero_variance)
else:
    print("No zero-variance features found.")

# ==============================
# 7. CORRELATION WITH TARGET
# ==============================

print("\n7. TOP FEATURES CORRELATED WITH SALEPRICE")
print("-" * 60)

correlations = train.corr(numeric_only=True)[target].drop(target)

correlations = correlations.abs().sort_values(ascending=False)

print(correlations.head(20).to_string())

# ==============================
# 8. ID CHECK
# ==============================

print("\n8. ID COLUMN")
print("-" * 60)

if "Id" in train.columns:
    print("Id exists.")

    print("Train Id minimum:", train["Id"].min())
    print("Train Id maximum:", train["Id"].max())

    print("Test Id minimum:", test["Id"].min())
    print("Test Id maximum:", test["Id"].max())

    print("Id correlation with Saleprice:",
          train["Id"].corr(train[target]))
else:
    print("Id column not found.")

# ==============================
# 9. FEATURE DATA TYPES
# ==============================

print("\n9. FEATURE DATA TYPES")
print("-" * 60)

print(train_features.dtypes.value_counts())

# ==============================
# 10. FINAL SUMMARY
# ==============================

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)

print("\nTrain shape:", train.shape)
print("Test shape :", test.shape)
print("Target     :", target)
print("Features   :", train_features.shape[1])
print("Missing values in target:", y.isnull().sum())
print("Zero variance features:", len(zero_variance))