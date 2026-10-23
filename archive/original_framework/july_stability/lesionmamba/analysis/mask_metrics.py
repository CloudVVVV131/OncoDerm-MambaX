from __future__ import annotations

import numpy as np


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / max(float(x.max() - x.min()), 1e-6)


def threshold_top_percent(x: np.ndarray, percent: float = 0.20) -> np.ndarray:
    x = normalize_map(x)
    thresh = np.quantile(x, 1.0 - percent)
    return x >= thresh


def saliency_mask_iou(saliency: np.ndarray, mask: np.ndarray, top_percent: float = 0.20) -> float:
    s = threshold_top_percent(saliency, top_percent)
    m = mask.astype(bool)
    inter = np.logical_and(s, m).sum()
    union = np.logical_or(s, m).sum()
    return float(inter / max(union, 1))


def pointing_game_hit(saliency: np.ndarray, mask: np.ndarray) -> float:
    y, x = np.unravel_index(np.argmax(saliency), saliency.shape)
    return float(mask.astype(bool)[y, x])


def lesion_background_ratio(saliency: np.ndarray, mask: np.ndarray) -> float:
    s = normalize_map(saliency)
    m = mask.astype(bool)
    lesion = s[m].mean() if m.any() else 0.0
    bg = s[~m].mean() if (~m).any() else 1e-6
    return float(lesion / max(bg, 1e-6))
