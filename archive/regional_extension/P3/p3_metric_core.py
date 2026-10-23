"""Unmodified pure metric arithmetic from current-direction FIM statistics."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from p3_common import OUT, dump as _dump
def dump(name, value):
    _dump(OUT / name, value)

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

def threshold(y, p, target):
    assert set(np.unique(y)) == {0, 1}
    order = np.argsort(-p, kind='stable')
    s, yy = (p[order], y[order])
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp = np.cumsum(yy)[ends]
    fp = ends + 1 - tp
    valid = np.flatnonzero(tp / y.sum() >= target)
    candidates = valid[fp[valid] == np.min(fp[valid])]
    index = candidates[np.argmax(s[ends[candidates]])]
    return float(s[ends[index]])

def confusion(y, p, t):
    pred = p >= t
    tp, fn = (int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 1))))
    fp, tn = (int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 0))))
    return {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn, 'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp)}

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

def tests():
    rng = np.random.Generator(np.random.PCG64(20261015))
    cases = []
    for i in range(20):
        y = np.tile([0, 1], 15)
        p = rng.random(30) if i % 2 else rng.integers(0, 6, 30) / 5
        w = rng.integers(0, 5, 30)
        w[:2] += 1
        got = GroupedMetric(y, p)(w)
        ref = np.array([roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w)])
        expanded = np.repeat(np.arange(len(y)), w)
        explicit = np.array([roc_auc_score(y[expanded], p[expanded]), average_precision_score(y[expanded], p[expanded])])
        err = float(max(abs(got - ref).max(), abs(got - explicit).max()))
        assert err < 1e-10
        for target in [0.9, 0.95]:
            brute = max(((confusion(y, p, t)['specificity'], t) for t in np.unique(p) if confusion(y, p, t)['sensitivity'] >= target))[1]
            assert threshold(y, p, target) == brute
        cases.append({'case': i, 'max_abs_error': err})
    patients = np.array([0, 0, 1, 2, 2, 3, 3, 3])
    multiplicities = np.bincount([0, 0, 2, 3], minlength=4)
    weights = multiplicities[patients]
    y = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    p = np.array([0.1, 0.8, 0.4, 0.8, 0.7, 0.2, 0.5, 0.5])
    assert np.allclose(GroupedMetric(y, p)(weights), [roc_auc_score(y, p, sample_weight=weights), average_precision_score(y, p, sample_weight=weights)], atol=1e-12)
    assert np.isnan(GroupedMetric(y, p)(np.zeros(len(y)))).all()
    dump('STATISTICAL_TESTS.json', {'status': 'PASSED', 'cases': cases, 'patient_repeat_weighting': True, 'threshold_ties_match_brute_force': True, 'single_class_draws_rejected': True, 'arithmetic_source': 'B2 b2_statistics.py GroupedMetric, threshold, confusion; no driver import'})
