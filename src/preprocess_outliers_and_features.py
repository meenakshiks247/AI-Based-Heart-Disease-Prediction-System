#!/usr/bin/env python3
"""
Preprocess heart disease dataset:
- Handle outliers in 'chol' and 'trestbps' using IQR capping
- Compute feature importance via absolute correlation with 'target'
- Save cleaned data to ../data/heart_cleaned.csv
"""
import os
import pandas as pd


def iqr_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def cap_outliers(series):
    lower, upper = iqr_bounds(series)
    return series.clip(lower=lower, upper=upper)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(base_dir, "..", "data", "heart.csv"))
    output_path = os.path.abspath(os.path.join(base_dir, "..", "data", "heart_cleaned.csv"))

    df = pd.read_csv(data_path)

    # Cap outliers using IQR bounds
    for col in ["chol", "trestbps"]:
        if col in df.columns:
            df[col] = cap_outliers(df[col])
        else:
            raise KeyError(f"Missing expected column: {col}")

    # Feature importance by absolute correlation with target
    if "target" not in df.columns:
        raise KeyError("Missing expected column: target")

    corr = df.corr(numeric_only=True)["target"].abs().sort_values(ascending=False)
    feature_importance = corr.drop("target")

    print("Feature importance (abs correlation with target):")
    for feature, score in feature_importance.items():
        print(f"{feature}: {score:.4f}")

    df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned dataset to: {output_path}")


if __name__ == "__main__":
    main()
