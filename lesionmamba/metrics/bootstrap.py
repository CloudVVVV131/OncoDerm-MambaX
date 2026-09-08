from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    iters: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    values = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        try:
            values.append(float(metric_fn(labels[idx], probs[idx])))
        except Exception:
            continue
    if not values:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    arr = np.array(values)
    return {
        "mean": float(np.nanmean(arr)),
        "ci_lower": float(np.nanpercentile(arr, 100 * alpha / 2)),
        "ci_upper": float(np.nanpercentile(arr, 100 * (1 - alpha / 2))),
    }
