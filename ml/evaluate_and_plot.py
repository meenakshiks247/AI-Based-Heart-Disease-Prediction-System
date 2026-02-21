from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

SEED = 42


def load_test_set(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and recreate the same test split used during training."""
    print(f"[INFO] Loading dataset from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at '{data_path}'.")

    df = pd.read_csv(data_path)
    if "target" not in df.columns:
        raise ValueError("Target column 'target' was not found in dataset.")

    X = df.drop(columns=["target"])
    y = df["target"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )
    print(f"[INFO] Test set prepared: {X_test.shape[0]} rows")
    return X_test, y_test


def get_roc_scores(model, X_test: pd.DataFrame) -> np.ndarray | None:
    """Get continuous scores for ROC; return None if unavailable."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return probs[:, 1]
        return probs
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return None


def save_confusion_plot(y_true: pd.Series, y_pred: np.ndarray, out_path: Path, model_name: str) -> None:
    """Save confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_roc_plot(y_true: pd.Series, y_scores: np.ndarray | None, out_path: Path, model_name: str) -> float:
    """Save ROC figure and return ROC-AUC (NaN if unavailable)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    roc_auc = float("nan")

    if y_scores is None:
        ax.text(0.5, 0.5, "ROC unavailable\n(no predict_proba/decision_function)", ha="center", va="center")
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


def print_best_summary(results_df: pd.DataFrame) -> None:
    """Print best model for key metrics."""
    for metric in ["accuracy", "f1", "roc_auc"]:
        valid = results_df.dropna(subset=[metric])
        if valid.empty:
            print(f"[SUMMARY] Best {metric}: unavailable")
            continue
        best = valid.loc[valid[metric].idxmax()]
        print(f"[SUMMARY] Best {metric}: {best['model_name']} ({best[metric]:.4f})")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "heart_cleaned.csv"
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_test_set(data_path)

    model_files = sorted(model_dir.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No trained model files found in '{model_dir}'.")

    results: list[dict[str, float | str]] = []
    for model_file in model_files:
        model_name = model_file.stem
        print(f"\n[INFO] Evaluating model: {model_name}")
        model = joblib.load(model_file)

        y_pred = model.predict(X_test)
        y_scores = get_roc_scores(model, X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = save_roc_plot(
            y_true=y_test,
            y_scores=y_scores,
            out_path=model_dir / f"{model_name}_roc.png",
            model_name=model_name,
        )
        save_confusion_plot(
            y_true=y_test,
            y_pred=y_pred,
            out_path=model_dir / f"{model_name}_confusion.png",
            model_name=model_name,
        )

        print(f"[INFO] Saved: {model_dir / f'{model_name}_confusion.png'}")
        print(f"[INFO] Saved: {model_dir / f'{model_name}_roc.png'}")

        results.append(
            {
                "model_name": model_name,
                "accuracy": acc,
                "f1": f1,
                "roc_auc": roc_auc,
            }
        )

    results_df = pd.DataFrame(results)
    leaderboard_path = model_dir / "roc_leaderboard.png"
    save_leaderboard(results_df, leaderboard_path)
    print(f"\n[INFO] Saved leaderboard: {leaderboard_path}")
    print_best_summary(results_df)


if __name__ == "__main__":
    main()
