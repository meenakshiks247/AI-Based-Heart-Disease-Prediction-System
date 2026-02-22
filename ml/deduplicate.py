from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("ml/data/heart_cleaned.csv")
OUT_PATH = Path("ml/data/heart_cleaned_dedup.csv")
WARN_PATH = Path("ml/models/dedup_warning.txt")
TARGET_COL = "target"


def class_distribution(y: pd.Series) -> pd.Series:
    """Return normalized class distribution."""
    return y.value_counts(normalize=True).sort_index()


def pick_most_representative_indices(
    df: pd.DataFrame,
    group_keys: pd.Series,
    seed: int,
) -> pd.Index:
    """
    Keep one row per group: row closest to class centroid (numeric feature space).
    Falls back to first row when centroid distance cannot be computed.
    """
    rng = np.random.default_rng(seed)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET_COL]

    # Precompute per-class centroids for numeric columns.
    centroids: dict[object, pd.Series] = {}
    if numeric_cols and TARGET_COL in df.columns:
        for cls, cls_df in df.groupby(TARGET_COL):
            centroids[cls] = cls_df[numeric_cols].mean(numeric_only=True)

    chosen_indices: list[int] = []
    for _, idxs in group_keys.groupby(group_keys).groups.items():
        group_df = df.loc[idxs]
        if len(group_df) == 1:
            chosen_indices.append(int(group_df.index[0]))
            continue

        if not numeric_cols or TARGET_COL not in df.columns:
            chosen_indices.append(int(group_df.index[0]))
            continue

        cls_mode = group_df[TARGET_COL].mode(dropna=False)
        cls_value = cls_mode.iloc[0] if not cls_mode.empty else group_df[TARGET_COL].iloc[0]
        centroid = centroids.get(cls_value)

        if centroid is None:
            chosen_indices.append(int(group_df.index[0]))
            continue

        num = group_df[numeric_cols].copy()
        # Fill NaNs with centroid so distance remains defined.
        num = num.fillna(centroid)
        distances = np.sqrt(((num - centroid) ** 2).sum(axis=1))
        min_distance = distances.min()
        tied = distances[distances == min_distance].index
        if len(tied) > 1:
            chosen_indices.append(int(rng.choice(tied.to_numpy())))
        else:
            chosen_indices.append(int(tied[0]))

    return pd.Index(chosen_indices)


def deduplicate(df: pd.DataFrame, strategy: str, seed: int) -> tuple[pd.DataFrame, str, int]:
    """
    Deduplicate dataframe by id column when available, otherwise exact duplicate rows.
    Returns deduped_df, dedup_mode, removed_count.
    """
    id_col = "patient_id" if "patient_id" in df.columns else ("id" if "id" in df.columns else None)

    if id_col is not None:
        dedup_mode = f"id-column ({id_col})"
        group_keys = df[id_col].astype(str).fillna("__NA__")
    else:
        dedup_mode = "exact-row"
        row_sig = pd.util.hash_pandas_object(df, index=False).astype(str)
        group_keys = row_sig

    if strategy == "keep-first":
        dedup_df = df.loc[~group_keys.duplicated(keep="first")].copy()
    elif strategy == "keep-random":
        rng = np.random.default_rng(seed)
        chosen = (
            df.assign(_g=group_keys, _r=rng.random(len(df)))
            .sort_values(["_g", "_r"])
            .drop_duplicates(subset="_g", keep="first")
            .drop(columns=["_g", "_r"])
        )
        dedup_df = chosen.copy()
    else:  # keep-most-representative
        chosen_idx = pick_most_representative_indices(df, group_keys, seed=seed)
        dedup_df = df.loc[chosen_idx].copy()

    dedup_df = dedup_df.sort_index().reset_index(drop=True)
    removed = len(df) - len(dedup_df)
    return dedup_df, dedup_mode, removed


def maybe_write_warning(pre_dist: pd.Series, post_dist: pd.Series, removed: int, strategy: str, mode: str) -> None:
    """Write warning log when any class proportion changes by >2%."""
    all_classes = sorted(set(pre_dist.index).union(set(post_dist.index)))
    pre = pre_dist.reindex(all_classes, fill_value=0.0)
    post = post_dist.reindex(all_classes, fill_value=0.0)
    deltas = (post - pre).abs()
    changed = deltas[deltas > 0.02]

    if changed.empty:
        print("[INFO] Class balance shift within 2% threshold for all classes.")
        return

    WARN_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Deduplication Warning",
        f"strategy={strategy}",
        f"mode={mode}",
        f"rows_removed={removed}",
        "Class proportion changes (>2%):",
    ]
    for cls, delta in changed.items():
        lines.append(f"class={cls}, delta={delta:.4f}, pre={pre.loc[cls]:.4f}, post={post.loc[cls]:.4f}")

    WARN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[WARN] Class balance changed by >2% for {len(changed)} class(es).")
    print(f"[WARN] Wrote warning log: {WARN_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate cleaned heart dataset.")
    parser.add_argument(
        "--strategy",
        choices=["keep-first", "keep-random", "keep-most-representative"],
        default="keep-first",
        help="Deduplication strategy.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random/tie-breaking strategies.")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"[ERROR] Input file not found: {DATA_PATH}")
        return 2

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded: {DATA_PATH} ({len(df)} rows)")

    if TARGET_COL not in df.columns:
        print(f"[ERROR] Required target column '{TARGET_COL}' not found.")
        return 2

    pre_dist = class_distribution(df[TARGET_COL])
    print("[INFO] Pre-dedup class distribution:")
    print(pre_dist.to_string())

    dedup_df, mode, removed = deduplicate(df, strategy=args.strategy, seed=args.seed)
    print(f"[INFO] Dedup mode: {mode}")
    print(f"[INFO] Strategy: {args.strategy}")
    print(f"[INFO] Rows removed: {removed}")
    print(f"[INFO] Rows after dedup: {len(dedup_df)}")

    post_dist = class_distribution(dedup_df[TARGET_COL])
    print("[INFO] Post-dedup class distribution:")
    print(post_dist.to_string())

    maybe_write_warning(pre_dist, post_dist, removed=removed, strategy=args.strategy, mode=mode)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dedup_df.to_csv(OUT_PATH, index=False)
    print(f"[INFO] Saved deduplicated file: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
