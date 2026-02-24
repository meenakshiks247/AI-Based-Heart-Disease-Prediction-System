"""Remove duplicate patient rows safely from the heart disease dataset.

Source  : ml/data/heart_cleaned.csv        (NOT modified)
Output  : ml/data/heart_cleaned_unique.csv (cleaned)
Audit   : ml/data/duplicates_removed.csv   (removed rows)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "heart_cleaned.csv"
OUTPUT_PATH = BASE_DIR / "data" / "heart_cleaned_unique.csv"
AUDIT_PATH = BASE_DIR / "data" / "duplicates_removed.csv"
TARGET_COL = "target"


def print_class_distribution(label: str, df: pd.DataFrame) -> None:
    """Print target class distribution with counts and percentages."""
    counts = df[TARGET_COL].value_counts().sort_index()
    total = len(df)
    print(f"[INFO] {label}:")
    for cls, count in counts.items():
        pct = count / total * 100
        print(f"         class {cls}: {count} ({pct:.1f}%)")


def main() -> None:
    # ── Load ─────────────────────────────────────────────────────────
    if not INPUT_PATH.exists():
        print(f"[WARN] Source file not found: {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    total_before = len(df)
    print(f"[INFO] Loaded {total_before} rows from {INPUT_PATH.name}")

    if TARGET_COL not in df.columns:
        print(f"[WARN] Target column '{TARGET_COL}' not found. Aborting.")
        return

    # ── Class distribution BEFORE cleaning ───────────────────────────
    print_class_distribution("Class distribution BEFORE cleaning", df)

    # ── Detect duplicates (all columns, keep first occurrence) ───────
    dup_mask = df.duplicated(keep="first")
    n_duplicates = dup_mask.sum()
    print(f"\n[INFO] Total rows before cleaning : {total_before}")
    print(f"[INFO] Duplicate rows detected    : {n_duplicates}")

    if n_duplicates == 0:
        print("[INFO] No duplicates found. Nothing to remove.")
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"[DONE] Saved unchanged dataset to {OUTPUT_PATH.name}")
        return

    # ── Separate duplicates and unique rows ──────────────────────────
    duplicates_df = df[dup_mask]
    unique_df = df[~dup_mask]

    print(f"[INFO] Unique rows remaining      : {len(unique_df)}")

    # ── Save duplicates for audit (do NOT shuffle) ───────────────────
    duplicates_df.to_csv(AUDIT_PATH, index=False)
    print(f"\n[INFO] Saved {len(duplicates_df)} duplicate rows to {AUDIT_PATH.name} (audit)")

    # ── Save cleaned dataset (do NOT shuffle) ────────────────────────
    unique_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Saved {len(unique_df)} unique rows to {OUTPUT_PATH.name}")

    # ── Class distribution AFTER cleaning ────────────────────────────
    print()
    print_class_distribution("Class distribution AFTER cleaning", unique_df)

    # ── Safety check: target column unchanged ────────────────────────
    if set(unique_df[TARGET_COL].unique()) == set(df[TARGET_COL].unique()):
        print(f"\n[INFO] Target column '{TARGET_COL}' values preserved correctly.")
    else:
        print(f"\n[WARN] Target column '{TARGET_COL}' values changed after cleaning!")

    # ── Confirm original file is untouched ───────────────────────────
    print(f"[INFO] Original file NOT modified: {INPUT_PATH.name}")
    print(f"\n[DONE] Deduplication complete.")


if __name__ == "__main__":
    main()
