# Script 2: Data Cleaning and Preprocessing
# Week 2 - ADA Global Academy SIWES Project
# Handles missing values, duplicates, encoding, and normalization

import pandas as pd
import numpy as np
import os

# ── Load Data ────────────────────────────────────────────────────────────────

def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student_data.csv")

    if not os.path.exists(data_path):
        print("Data file not found. Please run 01_scrape_data.py first.")
        return None

    df = pd.read_csv(data_path, sep=";")
    print(f"Data loaded. Shape: {df.shape}")
    return df


# ── Step 1: Inspect for Missing Values ──────────────────────────────────────

def check_missing_values(df):
    print("\n" + "=" * 50)
    print("MISSING VALUE CHECK")
    print("=" * 50)

    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100

    missing_report = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct.round(2)
    })

    print(missing_report[missing_report["Missing Count"] > 0])

    if missing.sum() == 0:
        print("No missing values found. Dataset is clean.")
    else:
        print(f"\nTotal missing values: {missing.sum()}")

    return df


# ── Step 2: Remove Duplicates ────────────────────────────────────────────────

def remove_duplicates(df):
    print("\n" + "=" * 50)
    print("DUPLICATE CHECK")
    print("=" * 50)

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    print(f"Rows before: {before} | Rows after: {after} | Removed: {before - after}")
    return df


# ── Step 3: Handle Missing Values ────────────────────────────────────────────

def handle_missing_values(df):
    print("\n" + "=" * 50)
    print("HANDLING MISSING VALUES")
    print("=" * 50)

    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled '{col}' with median: {median_val}")

    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  Filled '{col}' with mode: {mode_val}")

    print("Missing value handling complete.")
    return df


# ── Step 4: Encode Categorical Variables ─────────────────────────────────────

def encode_categorical(df):
    print("\n" + "=" * 50)
    print("ENCODING CATEGORICAL VARIABLES")
    print("=" * 50)

    # Binary encoding for yes/no and M/F columns
    binary_map = {
        "yes": 1, "no": 0,
        "F": 0, "M": 1,
        "U": 1, "R": 0,        # Urban / Rural
        "T": 1, "A": 0,        # Together / Apart (parent status)
        "GT3": 1, "LE3": 0,    # Family size
        "GP": 0, "MS": 1       # School
    }

    for col in df.select_dtypes(include=["object"]).columns:
        unique_vals = df[col].unique()
        if all(v in binary_map for v in unique_vals):
            df[col] = df[col].map(binary_map)
            print(f"  Binary encoded: '{col}'")
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            print(f"  One-hot encoded: '{col}'")

    print("Encoding complete.")
    return df


# ── Step 5: Normalize Numeric Columns ────────────────────────────────────────

def normalize_data(df):
    print("\n" + "=" * 50)
    print("NORMALIZATION")
    print("=" * 50)

    # Normalize only feature columns, not target (G3) or grade columns
    cols_to_normalize = ["age", "absences", "studytime", "Medu", "Fedu"]
    cols_to_normalize = [c for c in cols_to_normalize if c in df.columns]

    for col in cols_to_normalize:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df[col + "_norm"] = (df[col] - min_val) / (max_val - min_val)
            print(f"  Normalized: '{col}' → '{col}_norm'")

    return df


# ── Step 6: Save Cleaned Data ─────────────────────────────────────────────────

def save_clean_data(df):
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "student_data_clean.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Cleaned data saved to: {output_path}")
    print(f"   Final shape: {df.shape}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    if df is not None:
        df = check_missing_values(df)
        df = remove_duplicates(df)
        df = handle_missing_values(df)
        df = encode_categorical(df)
        df = normalize_data(df)
        save_clean_data(df)

        print("\n✅ Data cleaning complete. Proceed to 03_eda_visualization.py")
