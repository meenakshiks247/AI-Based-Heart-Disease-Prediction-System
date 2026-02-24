"""Prepare the Kaggle Cardiovascular Disease dataset for model training.

Source : ml/data/external/cardio_train.csv   (semicolon-separated)
Output : ml/data/cardio_cleaned.csv          (comma-separated)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "external" / "cardio_train.csv"
OUTPUT_PATH = BASE_DIR / "data" / "cardio_cleaned.csv"


def main() -> None:
    # ── Load ─────────────────────────────────────────────────────────
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Source file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, sep=";")
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── Drop id ──────────────────────────────────────────────────────
    df = df.drop(columns=["id"])

    # ── Convert age from days to years ───────────────────────────────
    df["age"] = (df["age"] / 365).round(1)
    print(f"[INFO] Converted age to years (range: {df['age'].min()} - {df['age'].max()})")

    # ── Remove exact duplicate rows ──────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    print(f"[INFO] Removed {before - len(df)} duplicate rows")

    # ── Remove unrealistic blood pressure ────────────────────────────
    before = len(df)
    df = df[(df["ap_hi"] > 0) & (df["ap_lo"] > 0)]       # no negative/zero
    df = df[df["ap_hi"] <= 250]
    df = df[df["ap_lo"] <= 200]
    df = df[df["ap_lo"] <= df["ap_hi"]]                   # diastolic <= systolic
    print(f"[INFO] Removed {before - len(df)} rows with unrealistic blood pressure")

    # ── Remove unrealistic height / weight ───────────────────────────
    before = len(df)
    df = df[(df["height"] >= 100) & (df["height"] <= 220)]  # cm
    df = df[(df["weight"] >= 30) & (df["weight"] <= 250)]   # kg
    print(f"[INFO] Removed {before - len(df)} rows with unrealistic height/weight")

    # ── Remove unrealistic age ───────────────────────────────────────
    before = len(df)
    df = df[(df["age"] >= 18) & (df["age"] <= 100)]
    print(f"[INFO] Removed {before - len(df)} rows with unrealistic age")

    # ── Outlier capping (IQR) on continuous columns ──────────────────
    continuous = ["age", "height", "weight", "ap_hi", "ap_lo"]
    for col in continuous:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        before_cap = ((df[col] < q1) | (df[col] > q99)).sum()
        df[col] = df[col].clip(lower=q1, upper=q99)
        if before_cap > 0:
            print(f"[INFO] Capped {before_cap} outliers in '{col}' to [{q1}, {q99}]")

    # ── Rename columns ───────────────────────────────────────────────
    df = df.rename(columns={
        "gender": "sex",
        "ap_hi": "systolic_bp",
        "ap_lo": "diastolic_bp",
        "cardio": "target",
    })

    # ── Ensure all numeric ───────────────────────────────────────────
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    df = df.reset_index(drop=True)

    # ── Save ─────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[INFO] Saved {len(df)} rows to {OUTPUT_PATH}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n[RESULT] Rows remaining : {len(df)}")
    print(f"[RESULT] Columns        : {df.columns.tolist()}")
    print(f"\n[RESULT] Class distribution (target):")
    print(df["target"].value_counts().to_string())


if __name__ == "__main__":
    main()
