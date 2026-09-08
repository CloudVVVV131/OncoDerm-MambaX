from __future__ import annotations

import numpy as np


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15, binary: bool = False) -> float:
    if binary:
        conf = probs.reshape(-1)
        true = labels.astype(int)
    else:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        true = labels
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (conf >= lo if idx == 0 else conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        acc = true[mask].mean() if binary else (pred[mask] == true[mask]).mean()
        avg_conf = conf[mask].mean()
        ece += mask.mean() * abs(acc - avg_conf)
    return float(ece)


def multiclass_brier(labels: np.ndarray, probs: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def calibration_curve_bins(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
    binary: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if binary:
        conf = probs.reshape(-1)
        true = labels.astype(int)
        pred = None
    else:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        true = labels
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (conf >= lo if idx == 0 else conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        xs.append(float(conf[mask].mean()))
        ys.append(float(true[mask].mean() if binary else (pred[mask] == true[mask]).mean()))
    return np.array(xs), np.array(ys)
