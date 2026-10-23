from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from lesionmamba.utils.io import ensure_dir


def save_confusion_matrix(cm: list[list[int]], labels: list[str], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    plt.figure(figsize=(7, 6))
    sns.heatmap(np.array(cm), annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close()


def save_reliability_diagram(confidence: np.ndarray, accuracy: np.ndarray, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.plot(confidence, accuracy, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close()


def save_binary_roc_pr(
    labels: np.ndarray,
    scores: np.ndarray,
    path_prefix: str | Path,
    title: str,
) -> None:
    prefix = Path(path_prefix)
    ensure_dir(prefix.parent)
    if len(np.unique(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix.with_name(prefix.name + "_roc.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix.with_name(prefix.name + "_pr.png"), dpi=200)
    plt.close()


def save_efficiency_tradeoff(df, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    required = {"latency_ms_b1", "Macro-F1", "Model"}
    if df.empty or not required.issubset(df.columns):
        return
    plot_df = df.dropna(subset=["latency_ms_b1", "Macro-F1"])
    if plot_df.empty:
        return
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=plot_df, x="latency_ms_b1", y="Macro-F1", hue="Model", s=80)
    for _, row in plot_df.iterrows():
        plt.text(row["latency_ms_b1"], row["Macro-F1"], str(row["Model"]), fontsize=8)
    plt.xlabel("Latency B=1 (ms/image)")
    plt.ylabel("Macro-F1")
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close()
