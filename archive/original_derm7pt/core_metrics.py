"""AST-extracted pure arithmetic from P4; no P4 imports or drivers."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

class GroupedMetric:

    def __init__(self, y, score):
        self.order = np.argsort(-score, kind='mergesort')
        s = score[self.order]
        self.starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
        self.y = y[self.order].astype(np.float64)

    def __call__(self, weights):
        w = weights[self.order]
        positive = np.add.reduceat(w * self.y, self.starts)
        negative = np.add.reduceat(w * (1 - self.y), self.starts)
        p, n = (positive.sum(), negative.sum())
        if p <= 0 or n <= 0:
            return np.array([np.nan, np.nan])
        tp, fp = (np.cumsum(positive), np.cumsum(negative))
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
        return np.array([np.sum(negative * (tp - 0.5 * positive)) / (p * n), np.sum(positive * precision) / p])

def ece(y, p):
    bins = np.minimum((p * 15).astype(int), 14)
    total = 0.0
    rows = []
    for k in range(15):
        ix = bins == k
        n = int(ix.sum())
        mean = float(p[ix].mean()) if n else None
        rate = float(y[ix].mean()) if n else None
        if n:
            total += n / len(y) * abs(mean - rate)
        rows.append({'bin': k, 'lower': k / 15, 'upper': (k + 1) / 15, 'n': n, 'mean_probability': mean, 'observed_rate': rate})
    return (total, rows)

def binary(y, p):
    clipped = np.clip(p, 1e-07, 1 - 1e-07)
    calibration, bins = ece(y, p)
    return ({'roc_auc': float(roc_auc_score(y, p)), 'average_precision': float(average_precision_score(y, p)), 'brier': float(np.mean((p - y) ** 2)), 'binary_nll': float(np.mean(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped))), 'ece15': calibration, 'images': len(y), 'positive_images': int(y.sum())}, bins)
