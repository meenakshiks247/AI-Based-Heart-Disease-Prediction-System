"""Generate a professional grouped-bar accuracy comparison for all trained models.

Reads:
    ml/models/model_results_holdout.csv   (Cleveland dataset)
    ml/models/cardio_results.csv          (Cardiovascular dataset)

Saves:
    ml/models/accuracy_dataset_comparison.png  (dpi 300)

Usage:
    python ml/plot_accuracy_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
HOLDOUT_CSV = BASE_DIR / "models" / "model_results_holdout.csv"
CARDIO_CSV = BASE_DIR / "models" / "cardio_results.csv"
OUTPUT_PNG = BASE_DIR / "models" / "accuracy_dataset_comparison.png"

# ── Load & normalise ─────────────────────────────────────────────────

df_cleveland = pd.read_csv(HOLDOUT_CSV)
# The Cleveland file stores accuracy as "holdout_accuracy"
if "holdout_accuracy" in df_cleveland.columns:
    df_cleveland = df_cleveland.rename(columns={"holdout_accuracy": "accuracy"})
df_cleveland = df_cleveland[["model_name", "accuracy"]].copy()
df_cleveland["dataset"] = "Cleveland Dataset"

df_cardio = pd.read_csv(CARDIO_CSV)
df_cardio = df_cardio[["model_name", "accuracy"]].copy()
df_cardio["dataset"] = "Cardiovascular Dataset"

# Combine
df = pd.concat([df_cleveland, df_cardio], ignore_index=True)

# Pivot so each row is a model, columns are datasets
pivot = df.pivot_table(index="model_name", columns="dataset", values="accuracy")
pivot = pivot.sort_index()

datasets = ["Cleveland Dataset", "Cardiovascular Dataset"]
for ds in datasets:
    if ds not in pivot.columns:
        pivot[ds] = 0.0
pivot = pivot.fillna(0.0)

# Keep only models that appear in BOTH datasets
pivot = pivot[(pivot[datasets[0]] > 0) & (pivot[datasets[1]] > 0)]

models = pivot.index.tolist()
n_models = len(models)
print(f"[INFO] Common models across both datasets: {models}")

# ── Best model per dataset ───────────────────────────────────────────

best = {}
for ds in datasets:
    col = pivot[ds]
    present = col[col > 0]
    if not present.empty:
        best[ds] = present.idxmax()

# ── Plot ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(n_models)
bar_width = 0.35
colors = {"Cleveland Dataset": "#4C72B0", "Cardiovascular Dataset": "#DD8452"}

for i, ds in enumerate(datasets):
    values = pivot[ds].values
    bars = ax.bar(x + i * bar_width, values, bar_width, label=ds,
                  color=colors[ds], edgecolor="white", linewidth=0.6)

    # Value labels above bars
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

# Annotate best models
for ds in datasets:
    if ds in best:
        model_idx = models.index(best[ds])
        col_offset = datasets.index(ds) * bar_width
        val = pivot.loc[best[ds], ds]
        ax.annotate(
            f"★ Best",
            xy=(model_idx + col_offset, val + 0.005),
            xytext=(0, 18), textcoords="offset points",
            ha="center", fontsize=8, fontweight="bold",
            color=colors[ds],
            arrowprops=dict(arrowstyle="-", color=colors[ds], lw=0.8),
        )

# ── Axes & styling ───────────────────────────────────────────────────

ax.set_title("Accuracy Comparison — Clinical vs Population Models",
             fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Model Names", fontsize=12, labelpad=8)
ax.set_ylabel("Accuracy Score (0–1)", fontsize=12, labelpad=8)
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(fontsize=10, loc="upper right")

fig.tight_layout()

# ── Save & show ──────────────────────────────────────────────────────

fig.savefig(OUTPUT_PNG, dpi=300)
print(f"[DONE] Accuracy comparison graph saved → {OUTPUT_PNG}")

plt.show()
