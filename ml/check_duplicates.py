from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


DATA_PATH = Path("ml/data/heart_cleaned.csv")
SAMPLES_OUT = Path("ml/models/duplicate_samples.csv")
DEDUP_OUT = Path("ml/data/heart_cleaned_dedup.csv")


def get_duplicate_mask(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for rows that are part of an exact duplicate group."""
    return df.duplicated(keep=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check exact duplicates in cleaned heart dataset.")
    parser.add_argument("--drop", action="store_true", help="Write deduplicated CSV to ml/data/heart_cleaned_dedup.csv")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"[ERROR] Dataset file not found: {DATA_PATH}")
        return 2

    print(f"[INFO] Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    total_rows = len(df)

    dup_mask = get_duplicate_mask(df)
    dup_rows = df[dup_mask]
    exact_duplicate_rows = int(df.duplicated().sum())

    print(f"[INFO] Total rows: {total_rows}")
    print(f"[INFO] Exact duplicate rows (excluding first in each group): {exact_duplicate_rows}")
    print(f"[INFO] Rows participating in duplicate groups: {len(dup_rows)}")

    if dup_rows.empty:
        print("[INFO] No duplicate groups found.")
    else:
        # Duplicate groups and counts by full-row value
        grouped_counts = dup_rows.groupby(list(df.columns), dropna=False).size().reset_index(name="duplicate_count")
        grouped_counts = grouped_counts.sort_values("duplicate_count", ascending=False)

        print("\n[INFO] Top 5 duplicated row values (full rows + duplicate_count):")
        print(grouped_counts.head(5).to_string(index=False))

        # Save one example per duplicate group with duplicate_count
        SAMPLES_OUT.parent.mkdir(parents=True, exist_ok=True)
        grouped_counts.to_csv(SAMPLES_OUT, index=False)
        print(f"\n[INFO] Saved duplicate samples: {SAMPLES_OUT}")

    if args.drop:
        dedup_df = df.drop_duplicates(keep="first")
        removed = len(df) - len(dedup_df)
        DEDUP_OUT.parent.mkdir(parents=True, exist_ok=True)
        dedup_df.to_csv(DEDUP_OUT, index=False)
        print(f"[INFO] Wrote deduplicated file: {DEDUP_OUT}")
        print(f"[INFO] Rows removed: {removed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
