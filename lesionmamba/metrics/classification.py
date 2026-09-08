from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)

from lesionmamba.data.datasets import ID_TO_LABEL, LABEL_MAP


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def multiclass_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    preds = probs.argmax(axis=1)
    metrics: dict[str, Any] = {}
    metrics["accuracy"] = float(accuracy_score(labels, preds))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, preds))
    metrics["macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(labels, preds, average="weighted", zero_division=0))
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(7)), zero_division=0
    )
    metrics["per_class"] = {
        ID_TO_LABEL[i]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(7)
    }
    onehot = np.eye(7)[labels]
    per_auc = {}
    for i in range(7):
        try:
            per_auc[ID_TO_LABEL[i]] = float(roc_auc_score(onehot[:, i], probs[:, i]))
        except ValueError:
            per_auc[ID_TO_LABEL[i]] = float("nan")
    valid = [v for v in per_auc.values() if not np.isnan(v)]
    metrics["per_class_auc"] = per_auc
    metrics["macro_auc"] = float(np.mean(valid)) if valid else float("nan")
    mel_idx = LABEL_MAP["mel"]
    try:
        metrics["mel_auc"] = float(roc_auc_score((labels == mel_idx).astype(int), probs[:, mel_idx]))
    except ValueError:
        metrics["mel_auc"] = float("nan")
    metrics["mel_sensitivity"] = float(recall_score(labels == mel_idx, preds == mel_idx, zero_division=0))
    metrics["confusion_matrix"] = confusion_matrix(labels, preds, labels=list(range(7))).tolist()
    return metrics
