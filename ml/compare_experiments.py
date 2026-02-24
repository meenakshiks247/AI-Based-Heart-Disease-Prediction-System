"""Compare heart-disease (small) vs cardiovascular (big-data) experiment results.

Loads both leaderboards, adds a dataset label, and produces a grouped bar
chart with two subplots: Accuracy and ROC AUC.

Output : ml/models/experiment_bigdata_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

SAFE_CSV = MODEL_DIR / "model_results_safe.csv"
CARDIO_CSV = MODEL_DIR / "cardio_results.csv"
OUTPUT_PNG = MODEL_DIR / "experiment_bigdata_comparison.png"


def load_and_tag(path: Path, dataset_label: str) -> pd.DataFrame:
    """Load a results CSV and add a 'dataset' column."""
    df = pd.read_csv(path)
    df["dataset"] = dataset_label
    return df


def main() -> None:
    if not SAFE_CSV.exists():
        raise FileNotFoundError(f"Not found: {SAFE_CSV}")
    if not CARDIO_CSV.exists():
        raise FileNotFoundError(f"Not found: {CARDIO_CSV}")

    heart = load_and_tag(SAFE_CSV, "Heart-Disease (1K)")
    cardio = load_and_tag(CARDIO_CSV, "Cardiovascular (68K)")

    # Keep only models present in BOTH experiments for a fair comparison
    common_models = sorted(
        set(heart["model_name"]) & set(cardio["model_name"])
    )
    if not common_models:
        # Fall back to showing all models side by side
        combined = pd.concat([heart, cardio], ignore_index=True)
        common_models = sorted(combined["model_name"].unique())
    else:
        combined = pd.concat([
            heart[heart["model_name"].isin(common_models)],
            cardio[cardio["model_name"].isin(common_models)],
        ], ignore_index=True)

    datasets = combined["dataset"].unique()
    n_models = len(common_models)
    x = np.arange(n_models)
    bar_width = 0.35
    colours = {datasets[0]: "#7f5af0", datasets[1]: "#2cb67d"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        "Experiment Comparison: Small Dataset vs Big Data",
        fontsize=15, fontweight="bold", y=1.01,
    )

    for ax, metric, title in zip(
        axes,
        ["accuracy", "roc_auc"],
        ["Accuracy", "ROC AUC"],
    ):
        for i, ds in enumerate(datasets):
            subset = combined[combined["dataset"] == ds]
            # Align to common_models order
            vals = [
                subset.loc[subset["model_name"] == m, metric].values[0]
                if m in subset["model_name"].values else 0
                for m in common_models
            ]
            offset = -bar_width / 2 + i * bar_width
            bars = ax.bar(
                x + offset, vals, bar_width,
                label=ds, color=colours[ds], edgecolor="white", linewidth=0.5,
            )
            # Value labels on bars
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("_", "\n") for m in common_models],
            fontsize=9,
        )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved comparison chart: {OUTPUT_PNG}")

    # ── Print combined table ─────────────────────────────────────────
    print("\n[INFO] Combined leaderboard:")
    cols = ["dataset", "model_name", "accuracy", "roc_auc", "f1"]
    print(combined[cols].to_string(index=False))


if __name__ == "__main__":
    main()
