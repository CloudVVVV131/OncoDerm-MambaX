"""2026-10-12: FULL-ZERO seed42 fixed-model statistics.
Bootstrap PCG64 20261015; 10000 common paired patient-cluster draws.
GroupedMetric and threshold arithmetic follow tested B2 pure functions.
No calibration fitting or test/external threshold optimization.
"""
from pathlib import Path
import os
import sys
import json
import time
import hashlib
from datetime import datetime
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '2'
import numpy as np
import pandas as pd
from scipy.special import softmax, expit
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, f1_score, log_loss
from threadpoolctl import threadpool_limits

PRIVATE = OUT / 'private_predictions'
DEADLINE = datetime.fromisoformat('2026-10-13T06:12:40+08:00').timestamp()
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
UNITS = [('E_REAL', 'softmax_MEL', 0, 4), ('E_REAL', 'endpoint_MEL', 0, 7),
         ('E_NULL', 'softmax_MEL', 1, 4), ('E_NULL', 'endpoint_MEL', 1, 7)]


def dump(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


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
        p, n = positive.sum(), negative.sum()
        if p <= 0 or n <= 0:
            return np.array([np.nan, np.nan])
        tp, fp = np.cumsum(positive), np.cumsum(negative)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        return np.array([np.sum(negative * (tp - .5 * positive)) / (p * n), np.sum(positive * precision) / p])


def threshold(y, p, target):
    assert set(np.unique(y)) == {0, 1}
    order = np.argsort(-p, kind='stable')
    s, yy = p[order], y[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp = np.cumsum(yy)[ends]
    fp = ends + 1 - tp
    valid = np.flatnonzero(tp / y.sum() >= target)
    candidates = valid[fp[valid] == np.min(fp[valid])]
    index = candidates[np.argmax(s[ends[candidates]])]
    return float(s[ends[index]])


def confusion(y, p, t):
    pred = p >= t
    tp, fn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 1)))
    fp, tn = int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 0)))
    return {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
            'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp)}


def ece(y, p):
    bins = np.minimum((p * 15).astype(int), 14)
    total = 0.
    rows = []
    for k in range(15):
        ix = bins == k
        n = int(ix.sum())
        mean = float(p[ix].mean()) if n else None
        rate = float(y[ix].mean()) if n else None
        if n:
            total += n / len(y) * abs(mean - rate)
        rows.append({'bin': k, 'lower': k / 15, 'upper': (k + 1) / 15, 'n': n,
                     'mean_probability': mean, 'observed_rate': rate})
    return total, rows


def binary(y, p):
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    calibration, bins = ece(y, p)
    return {'roc_auc': float(roc_auc_score(y, p)), 'average_precision': float(average_precision_score(y, p)),
            'brier': float(np.mean((p - y)**2)),
            'binary_nll': float(np.mean(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped))),
            'ece15': calibration, 'images': len(y), 'positive_images': int(y.sum())}, bins


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
        for target in [.9, .95]:
            brute = max((confusion(y, p, t)['specificity'], t) for t in np.unique(p) if confusion(y, p, t)['sensitivity'] >= target)[1]
            assert threshold(y, p, target) == brute
        cases.append({'case': i, 'max_abs_error': err})
    patients = np.array([0, 0, 1, 2, 2, 3, 3, 3])
    multiplicities = np.bincount([0, 0, 2, 3], minlength=4)
    weights = multiplicities[patients]
    y = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    p = np.array([.1, .8, .4, .8, .7, .2, .5, .5])
    assert np.allclose(GroupedMetric(y, p)(weights), [roc_auc_score(y, p, sample_weight=weights), average_precision_score(y, p, sample_weight=weights)], atol=1e-12)
    assert np.isnan(GroupedMetric(y, p)(np.zeros(len(y)))).all()
    dump('STATISTICAL_TESTS.json', {'status': 'PASSED', 'cases': cases, 'patient_repeat_weighting': True,
         'threshold_ties_match_brute_force': True, 'single_class_draws_rejected': True,
         'arithmetic_source': 'B2 b2_statistics.py GroupedMetric, threshold, confusion; no driver import'})


def main():
    tests()
    assert json.loads((OUT / 'EVALUATION_STATUS.json').read_text())['status'] == 'COMPLETE'
    metadata = {k: pd.read_csv(PRIVATE / f'{k}_alignment.csv') for k in ['HAM_val', 'HAM_test', 'SIIM']}
    predictions = {}
    metrics, internal, thresholds, counts, rocs, prs, reliability = [], [], [], [], [], [], []
    for condition in ['FULL', 'ZERO']:
        for cohort in ['HAM_val', 'HAM_test', 'SIIM']:
            state = json.loads((PRIVATE / f'{condition}_{cohort}_status.json').read_text())
            assert state['status'] == 'COMPLETE' and state['next_index'] == len(metadata[cohort])
            logits = np.load(PRIVATE / f'{condition}_{cohort}_logits.npy', allow_pickle=False)
            p = np.load(PRIVATE / f'{condition}_{cohort}_probabilities.npy', allow_pickle=False).astype(np.float64)
            assert np.isfinite(logits).all() and np.isfinite(p).all()
            assert np.allclose(p[:, :, :7], softmax(logits[:, :, :7], axis=2), atol=2e-7, rtol=1e-6)
            assert np.allclose(p[:, :, 7], expit(logits[:, :, 7]), atol=2e-7, rtol=1e-6)
            predictions[(condition, cohort)] = p
            if cohort == 'SIIM':
                y = metadata[cohort].target.to_numpy()
                units = UNITS
            else:
                y7 = metadata[cohort].dx.map({v: k for k, v in enumerate(CLASSES)}).to_numpy()
                y = (y7 == 4).astype(int)
                units = [('NATIVE', 'softmax_MEL', 0, 4), ('NATIVE', 'endpoint_MEL', 0, 7)]
                prob = p[:, 0, :7]
                correct = (prob.argmax(1) == y7).astype(int)
                conf = prob.max(1)
                top_ece, _ = ece(correct, conf)
                internal.append({'condition': condition, 'cohort': cohort,
                    'macro_f1': float(f1_score(y7, prob.argmax(1), labels=list(range(7)), average='macro')),
                    'multiclass_brier_sum7': float(np.mean(np.sum((prob - np.eye(7)[y7])**2, axis=1))),
                    'multiclass_nll': float(np.mean(-np.log(np.clip(prob[np.arange(len(y7)), y7], 1e-7, 1)))),
                    'top_label_ece15': top_ece})
            for mode, head, c, s in units:
                score = p[:, c, s]
                tags = {'condition': condition, 'cohort': cohort, 'input': mode, 'head': head}
                m, bins = binary(y, score)
                metrics.append({**tags, **m})
                reliability.extend([{**tags, **b} for b in bins])
                fpr, tpr, th = roc_curve(y, score, drop_intermediate=False)
                rocs.extend([{**tags, 'point': i, 'fpr': a, 'tpr': b, 'threshold': t} for i, (a, b, t) in enumerate(zip(fpr, tpr, th))])
                precision, recall, th = precision_recall_curve(y, score, drop_intermediate=False)
                prs.extend([{**tags, 'point': i, 'recall': r, 'precision': pr, 'threshold': th[i] if i < len(th) else None}
                            for i, (r, pr) in enumerate(zip(recall, precision))])
                val = predictions[(condition, 'HAM_val')][:, 0, s]
                val_y = (metadata['HAM_val'].dx == 'mel').to_numpy(dtype=int)
                for target in [.9, .95]:
                    t = threshold(val_y, val, target)
                    if cohort == 'HAM_val':
                        thresholds.append({'condition': condition, 'head': head, 'target_sensitivity': target,
                                           'threshold': t, 'source': 'same_new_model_HAM_val_NATIVE', **confusion(val_y, val, t)})
                    counts.append({**tags, 'target_sensitivity': target, 'threshold': t, **confusion(y, score, t)})
    for name, rows in [('METRICS', metrics), ('INTERNAL_MULTICLASS_METRICS', internal), ('DEVELOPMENT_THRESHOLDS', thresholds),
                       ('THRESHOLD_CONFUSION_COUNTS', counts), ('ROC_COORDINATES', rocs), ('PR_COORDINATES', prs), ('RELIABILITY_BINS', reliability)]:
        pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False)
    siim = metadata['SIIM']
    y = siim.target.to_numpy()
    patients, cluster = np.unique(siim.patient_id.astype(str), return_inverse=True)
    assert len(patients) == 2056
    calculators = [[GroupedMetric(y, predictions[(condition, 'SIIM')][:, c, s]) for mode, head, c, s in UNITS] for condition in ['FULL', 'ZERO']]
    ones = np.ones(len(y))
    point = np.array([calculators[0][u](ones) - calculators[1][u](ones) for u in range(4)])
    for condition_index, condition in enumerate(['FULL', 'ZERO']):
        for u, (mode, head, c, s) in enumerate(UNITS):
            p = predictions[(condition, 'SIIM')][:, c, s]
            assert np.allclose(calculators[condition_index][u](ones), [roc_auc_score(y, p), average_precision_score(y, p)], atol=1e-12)
    rng = np.random.Generator(np.random.PCG64(20261015))
    draws = np.full((10000, 4, 2), np.nan)
    accepted, rejected = 0, 0
    checkpoint = OUT / 'BOOTSTRAP_PROGRESS.json'
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        accepted, rejected = state['accepted'], state['rejected']
        rng.bit_generator.state = state['rng_state']
        draws = np.load(OUT / 'bootstrap_differences_private.npy', allow_pickle=False)
    while accepted < 10000:
        counts = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))
        weights = counts[cluster].astype(float)
        result = np.array([calculators[0][u](weights) - calculators[1][u](weights) for u in range(4)])
        if not np.isfinite(result).all():
            rejected += 1
            assert rejected < 10000, 'Excessive degenerate draws'
            continue
        draws[accepted] = result
        accepted += 1
        if accepted % 250 == 0:
            np.save(OUT / 'bootstrap_differences_private.npy', draws, allow_pickle=False)
            dump('BOOTSTRAP_PROGRESS.json', {'accepted': accepted, 'rejected': rejected, 'rng_state': rng.bit_generator.state,
                 'target': 10000, 'seed': 20261015, 'unit': 'patient', 'paired': True})
            print(f'BOOTSTRAP {accepted}/10000', flush=True)
            if time.time() >= DEADLINE or (OUT / 'STOP_REQUESTED').exists():
                dump('STATISTICS_STATUS.json', {'status': 'PARTIAL', 'accepted': accepted})
                return
    intervals = np.quantile(draws, [.025, .975], axis=0)
    result_rows = []
    for u, (mode, head, c, s) in enumerate(UNITS):
        for m, metric in enumerate(['roc_auc', 'average_precision']):
            result_rows.append({'contrast': 'FULL_minus_ZERO', 'training_seed': 42, 'cohort': 'SIIM', 'input': mode,
                'head': head, 'metric': metric, 'difference': point[u, m], 'ci_low': intervals[0, u, m], 'ci_high': intervals[1, u, m],
                'role': 'PRIMARY' if mode == 'E_REAL' and head == 'softmax_MEL' and metric == 'average_precision' else 'SECONDARY_DESCRIPTIVE',
                'bootstrap_draws': 10000, 'cluster_unit': 'patient', 'clusters': 2056, 'CI': 'paired percentile 95%, fixed seed42 model pair'})
    pd.DataFrame(result_rows).to_csv(OUT / 'PAIRED_PATIENT_CLUSTER_CI.csv', index=False)
    dump('STATISTICS_STATUS.json', {'status': 'COMPLETE', 'accepted_draws': 10000, 'rejected_draws': rejected,
         'training_seeds': [42], 'primary': 'SIIM E_REAL softmax MEL average precision FULL-ZERO',
         'secondary_CIs': 'descriptive, no multiplicity-adjusted confirmatory claims',
         'uncertainty_scope': 'Patients conditional on two fixed seed42 fits, not three-seed algorithm uncertainty',
         'calibration_fitted': False, 'external_threshold_selection': False})


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        if '--tests-only' in sys.argv:
            tests()
        else:
            main()
