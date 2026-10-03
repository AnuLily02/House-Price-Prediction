import pandas as pd
import os

# File paths
train_path = "data/Cleaned train.csv"
test_path = "data/Cleaned test.csv"


def inspect_csv(file_path, name):
    print("\n" + "=" * 70)
    print(f"            {name}")
    print("=" * 70)

    if not os.path.exists(file_path):
        print(f"ERROR: File not found -> {file_path}")
        return

    df = pd.read_csv(file_path)

    # Basic information
    print("\n1. DATASET SHAPE")
    print("-" * 70)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # Column names
    print("\n2. COLUMN NAMES")
    print("-" * 70)
    for i, column in enumerate(df.columns, start=1):
        print(f"{i}. {column}")

    # First 5 rows
    print("\n3. FIRST 5 ROWS")
    print("-" * 70)
    print(df.head().to_string())

    # Data types
    print("\n4. DATA TYPES")
    print("-" * 70)
    print(df.dtypes.to_string())

    # Missing values
    print("\n5. MISSING VALUES")
    print("-" * 70)

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        print("No missing values found.")
    else:
        print(missing.to_string())

    # Duplicate rows
    print("\n6. DUPLICATE ROWS")
    print("-" * 70)
    print(f"Number of duplicate rows: {df.duplicated().sum()}")

    # Numerical columns
    print("\n7. NUMERICAL COLUMNS")
    print("-" * 70)
    numerical_columns = df.select_dtypes(include="number").columns.tolist()

    for column in numerical_columns:
        print(column)

    # Categorical columns
    print("\n8. CATEGORICAL COLUMNS")
    print("-" * 70)
    categorical_columns = df.select_dtypes(include="object").columns.tolist()

    if categorical_columns:
        for column in categorical_columns:
            print(column)
    else:
        print("No categorical columns found.")

    # Basic statistics
    print("\n9. BASIC STATISTICS")
    print("-" * 70)
    print(df.describe().to_string())

    # Unique values for categorical columns
    if categorical_columns:
        print("\n10. CATEGORICAL COLUMN DETAILS")
        print("-" * 70)

        for column in categorical_columns:
            print(f"\n{column}")
            print(f"Unique values: {df[column].nunique()}")

            # Show first 10 unique values only
            print(
                "Examples:",
                df[column].dropna().unique()[:10]
            )

    return df


# Inspect TRAIN dataset
train_df = inspect_csv(train_path, "TRAIN DATASET")

# Inspect TEST dataset
test_df = inspect_csv(test_path, "TEST DATASET")


# Compare train and test columns
if train_df is not None and test_df is not None:

    print("\n" + "=" * 70)
    print("            TRAIN vs TEST COMPARISON")
    print("=" * 70)

    train_columns = set(train_df.columns)
    test_columns = set(test_df.columns)

    print("\nColumns present in TRAIN but not TEST:")
    print(train_columns - test_columns)

    print("\nColumns present in TEST but not TRAIN:")
    print(test_columns - train_columns)

    print("\nCommon columns:")
    print(len(train_columns & test_columns))

    print("\nTrain shape:", train_df.shape)
    print("Test shape :", test_df.shape)