from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("ml/data/heart_cleaned_dedup.csv")
TRAIN_PATH = Path("ml/data/train.csv")
TEST_PATH = Path("ml/data/test.csv")
TARGET_COL = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ALLOWED_DIFF = 0.03  # 3%


def class_distribution(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Return normalized class distribution sorted by class value."""
    return df[target_col].value_counts(normalize=True).sort_index()


def print_distribution(label: str, dist: pd.Series) -> None:
    """Print class distribution in a readable format."""
    print(f"\n[INFO] {label}:")
    for cls, ratio in dist.items():
        print(f"  class {cls}: {ratio:.4f} ({ratio * 100:.2f}%)")


def max_distribution_shift(base: pd.Series, other: pd.Series) -> float:
    """Compute max absolute per-class shift between two distributions."""
    classes = sorted(set(base.index).union(set(other.index)))
    base_aligned = base.reindex(classes, fill_value=0.0)
    other_aligned = other.reindex(classes, fill_value=0.0)
    return float((base_aligned - other_aligned).abs().max())


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"[ERROR] Input file not found: {INPUT_PATH}")
        return 2

    df = pd.read_csv(INPUT_PATH)
    if TARGET_COL not in df.columns:
        print(f"[ERROR] Target column '{TARGET_COL}' not found in {INPUT_PATH}")
        return 2

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
    )

    # Save outputs
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    # Print distributions
    overall_dist = class_distribution(df, TARGET_COL)
    train_dist = class_distribution(train_df, TARGET_COL)
    test_dist = class_distribution(test_df, TARGET_COL)

    print(f"[INFO] Loaded rows: {len(df)}")
    print(f"[INFO] Saved train rows: {len(train_df)} -> {TRAIN_PATH}")
    print(f"[INFO] Saved test rows:  {len(test_df)} -> {TEST_PATH}")

    print_distribution("Overall class distribution", overall_dist)
    print_distribution("Train class distribution", train_dist)
    print_distribution("Test class distribution", test_dist)

    # Warn if class balance drifts by more than 3%
    train_shift = max_distribution_shift(overall_dist, train_dist)
    test_shift = max_distribution_shift(overall_dist, test_dist)
    if train_shift > MAX_ALLOWED_DIFF or test_shift > MAX_ALLOWED_DIFF:
        print(
            f"\n[WARN] Class distribution shift > 3% detected "
            f"(train max shift={train_shift:.4f}, test max shift={test_shift:.4f})."
        )
    else:
        print(
            f"\n[INFO] Class balance maintained within 3% "
            f"(train max shift={train_shift:.4f}, test max shift={test_shift:.4f})."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
