"""Generate a professional bar chart comparing holdout accuracy of all models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
HOLDOUT_CSV = BASE_DIR / "models" / "model_results_holdout.csv"
OUTPUT_PNG = BASE_DIR / "models" / "accuracy_comparison.png"


def main() -> None:
    if not HOLDOUT_CSV.exists():
        raise FileNotFoundError(f"Not found: {HOLDOUT_CSV}")

    df = pd.read_csv(HOLDOUT_CSV)
    df = df.sort_values(by="holdout_accuracy", ascending=False).reset_index(drop=True)

    models = df["model_name"].tolist()
    accuracies = df["holdout_accuracy"].tolist()
    best_idx = 0  # first after sort = highest accuracy

    # ── Colours ──────────────────────────────────────────────────────
    colours = ["#7f5af0" if i == best_idx else "#2cb67d" for i in range(len(models))]

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(models, accuracies, color=colours, edgecolor="white", linewidth=0.6)

    # Value labels above each bar
    for bar, val in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(
        "Holdout Accuracy Comparison \u2014 Heart Disease Models",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy (0\u20131)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Rotate x labels for readability
    plt.xticks(rotation=20, ha="right", fontsize=10)

    # Legend for highlight
    ax.bar([], [], color="#7f5af0", label="Best Model")
    ax.bar([], [], color="#2cb67d", label="Other Models")
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"[DONE] Saved accuracy comparison chart: {OUTPUT_PNG}")

    plt.show()


if __name__ == "__main__":
    main()
