from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

SEED = 42
TARGET_COL = "target"


# ── Data loading ─────────────────────────────────────────────────────


def load_test_set(test_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the held-out test split produced by the safe pipeline."""
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    df = pd.read_csv(test_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {test_path}")

    X_test = df.drop(columns=[TARGET_COL])
    y_test = df[TARGET_COL]
    print(f"[INFO] Loaded test set: {X_test.shape[0]} rows, {X_test.shape[1]} features")
    return X_test, y_test


# ── Model discovery ──────────────────────────────────────────────────


def resolve_model_list(
    model_dir: Path,
    override_names: list[str] | None = None,
) -> tuple[list[tuple[str, Path]], str]:
    """Return an ordered list of (display_name, joblib_path) pairs.

    Strategy:
        1. If *override_names* is given, use those names directly (try
           ``<name>_safe.joblib`` first, then ``<name>.joblib``).
        2. Otherwise read the canonical model list from the results CSV
           (safe preferred, legacy fallback).

    Returns:
        (model_list, source)  where *source* is ``"safe"`` or ``"legacy"``.
    """
    safe_csv = model_dir / "model_results_safe.csv"
    legacy_csv = model_dir / "model_results.csv"

    # Determine source
    if safe_csv.exists():
        source = "safe"
        print("[INFO] Using safe model results")
    elif legacy_csv.exists():
        source = "legacy"
        print("[WARN] Falling back to legacy model results")
    else:
        raise FileNotFoundError(
            f"Neither '{safe_csv.name}' nor '{legacy_csv.name}' found in {model_dir}.\n"
            "  Suggested fix: run  python ml/run_training.py  to train models first."
        )

    # Build name list
    if override_names:
        names = override_names
        print(f"[INFO] --models override: evaluating {names}")
    else:
        csv_path = safe_csv if source == "safe" else legacy_csv
        df = pd.read_csv(csv_path)
        if "model_name" not in df.columns:
            raise ValueError(f"'model_name' column missing in {csv_path}")
        names = df["model_name"].tolist()

    # Resolve to file paths
    models: list[tuple[str, Path]] = []
    for name in names:
        if source == "safe":
            candidates = [
                model_dir / f"{name}_safe.joblib",
                model_dir / f"{name}.joblib",
            ]
        else:
            candidates = [
                model_dir / f"{name}.joblib",
                model_dir / f"{name}_safe.joblib",
            ]

        resolved = None
        for c in candidates:
            if c.exists():
                resolved = c
                break

        if resolved is None:
            print(f"[WARN] Skipping missing: {candidates[0].name}")
        else:
            models.append((name, resolved))

    return models, source


# ── Plotting helpers ─────────────────────────────────────────────────


def get_roc_scores(model: object, X_test: pd.DataFrame) -> np.ndarray | None:
    """Get continuous scores for ROC; return None if unavailable."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return probs[:, 1]
        return probs
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return None


def save_confusion_plot(
    y_true: pd.Series, y_pred: np.ndarray, out_path: Path, model_name: str,
) -> None:
    """Save confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_roc_plot(
    y_true: pd.Series, y_scores: np.ndarray | None, out_path: Path, model_name: str,
) -> float:
    """Save ROC figure and return ROC-AUC (NaN if unavailable)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    roc_auc = float("nan")

    if y_scores is None:
        ax.text(0.5, 0.5, "ROC unavailable\n(no predict_proba/decision_function)",
                ha="center", va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"ROC Curve - {model_name}")
    else:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {model_name}")
        ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return roc_auc


def save_leaderboard(results_df: pd.DataFrame, out_path: Path) -> None:
    """Save ROC-AUC leaderboard bar chart."""
    plot_df = results_df.sort_values("roc_auc", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_df["model_name"], plot_df["roc_auc"])
    ax.set_title("Model Leaderboard by ROC-AUC")
    ax.set_xlabel("Model")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.05)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def print_summary_table(results_df: pd.DataFrame) -> None:
    """Print a concise summary table of evaluated models."""
    summary = (
        results_df
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    header = f"{'Model':<25} {'ROC AUC':>10} {'F1':>10} {'Accuracy':>10}"
    print(header)
    print("-" * len(header))
    for _, row in summary.iterrows():
        roc = f"{row['roc_auc']:.4f}" if pd.notna(row["roc_auc"]) else "   N/A"
        f1 = f"{row['f1']:.4f}" if pd.notna(row["f1"]) else "   N/A"
        acc = f"{row['accuracy']:.4f}" if pd.notna(row["accuracy"]) else "   N/A"
        print(f"{row['model_name']:<25} {roc:>10} {f1:>10} {acc:>10}")
    print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────


def is_estimator(obj: object) -> bool:
    """Check whether a loaded object is a usable estimator / pipeline."""
    return (
        hasattr(obj, "predict")
        or isinstance(obj, BaseEstimator)
        or hasattr(obj, "predict_proba")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained models and generate plots.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model names to evaluate (overrides CSV list).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    test_path = base_dir / "data" / "test.csv"
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    X_test, y_test = load_test_set(test_path)

    # Resolve model list from CSV (or --models override)
    override_names = [n.strip() for n in args.models.split(",")] if args.models else None
    model_list, source = resolve_model_list(model_dir, override_names)

    if not model_list:
        raise RuntimeError(
            "No valid model files to evaluate.\n"
            "  Suggested fix: run  python ml/run_training.py  to train models first."
        )

    # Evaluate each model
    results: list[dict[str, float | str]] = []
    for display_name, model_path in model_list:
        obj = joblib.load(model_path)

        if not is_estimator(obj):
            print(f"\n[DEBUG] Skipping non-estimator artifact: {model_path.name}")
            continue

        print(f"\n[INFO] Evaluating model: {display_name}  ({model_path.name})")

        y_pred = obj.predict(X_test)
        y_scores = get_roc_scores(obj, X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_val = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = save_roc_plot(
            y_true=y_test,
            y_scores=y_scores,
            out_path=model_dir / f"{display_name}_roc.png",
            model_name=display_name,
        )
        save_confusion_plot(
            y_true=y_test,
            y_pred=y_pred,
            out_path=model_dir / f"{display_name}_confusion.png",
            model_name=display_name,
        )

        print(f"[INFO] Saved: {display_name}_confusion.png")
        print(f"[INFO] Saved: {display_name}_roc.png")

        results.append(
            {
                "model_name": display_name,
                "accuracy": acc,
                "f1": f1_val,
                "roc_auc": roc_auc,
            }
        )

    results_df = pd.DataFrame(results)

    # Leaderboard chart
    leaderboard_path = model_dir / "roc_leaderboard.png"
    save_leaderboard(results_df, leaderboard_path)
    print(f"\n[INFO] Saved leaderboard: {leaderboard_path}")

    # Summary table
    print_summary_table(results_df)


if __name__ == "__main__":
    main()
