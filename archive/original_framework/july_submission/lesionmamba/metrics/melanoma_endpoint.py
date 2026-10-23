from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from lesionmamba.data.datasets import LABEL_MAP
from lesionmamba.metrics.calibration import expected_calibration_error


def binary_stats(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    tp = float(((pred == 1) & (y_true == 1)).sum())
    tn = float(((pred == 0) & (y_true == 0)).sum())
    fp = float(((pred == 1) & (y_true == 0)).sum())
    fn = float(((pred == 0) & (y_true == 1)).sum())
    sens = tp / max(tp + fn, 1.0)
    spec = tn / max(tn + fp, 1.0)
    return {"sensitivity": sens, "specificity": spec, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def threshold_youden(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([np.array([0.0, 1.0]), scores]))
    best_t, best_j = 0.5, -1.0
    for t in candidates:
        s = binary_stats(y_true, scores, float(t))
        j = s["sensitivity"] + s["specificity"] - 1.0
        if j > best_j:
            best_j = j
            best_t = float(t)
    return best_t


def threshold_for_specificity(y_true: np.ndarray, scores: np.ndarray, target_spec: float) -> float:
    candidates = np.unique(np.concatenate([np.array([0.0, 1.0]), scores]))
    best_t = 1.0
    best_sens = -1.0
    for t in candidates:
        s = binary_stats(y_true, scores, float(t))
        if s["specificity"] >= target_spec and s["sensitivity"] > best_sens:
            best_t = float(t)
            best_sens = s["sensitivity"]
    return best_t


def endpoint_metrics_from_probs(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    mel_idx = LABEL_MAP["mel"]
    return endpoint_metrics_from_scores(labels, probs[:, mel_idx], thresholds)


def endpoint_metrics_from_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    mel_idx = LABEL_MAP["mel"]
    y = (labels == mel_idx).astype(int)
    thresholds = thresholds or {
        "youden": threshold_youden(y, scores),
        "spec90": threshold_for_specificity(y, scores, 0.90),
        "spec95": threshold_for_specificity(y, scores, 0.95),
    }
    out: dict[str, Any] = {}
    out["roc_auc"] = float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan")
    out["pr_auc"] = float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else float("nan")
    out.update({f"youden_{k}": v for k, v in binary_stats(y, scores, thresholds["youden"]).items()})
    out["sensitivity"] = out["youden_sensitivity"]
    out["specificity"] = out["youden_specificity"]
    out["sens_at_90_spec"] = binary_stats(y, scores, thresholds["spec90"])["sensitivity"]
    out["sens_at_95_spec"] = binary_stats(y, scores, thresholds["spec95"])["sensitivity"]
    out["brier"] = float(brier_score_loss(y, scores))
    out["ece"] = float(expected_calibration_error(y, scores[:, None], n_bins=15, binary=True))
    return out, thresholds
