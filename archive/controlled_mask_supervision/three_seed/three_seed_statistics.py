"""Per-fit metrics first, then arithmetic means; common paired patient draws."""
from pathlib import Path
import os
import json
import time
import hashlib
import shutil
import ast
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import fim_statistics as frozen

OUT = Path(__file__).resolve().parent
WINDOW = json.loads((OUT / 'WINDOW.json').read_text(encoding='utf-8-sig'))
OLD = Path(WINDOW['source_seed42'])
SEEDS = [42, 2024, 3407]
CONDITIONS = ['FULL', 'ZERO']
TABLES = ['METRICS', 'INTERNAL_MULTICLASS_METRICS', 'DEVELOPMENT_THRESHOLDS',
          'THRESHOLD_CONFUSION_COUNTS', 'ROC_COORDINATES', 'PR_COORDINATES', 'RELIABILITY_BINS']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(name, obj):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, p)


def tests():
    old_tree = ast.parse((OLD / 'fim_statistics.py').read_text(encoding='utf-8'))
    new_tree = ast.parse((OUT / 'fim_statistics.py').read_text(encoding='utf-8'))
    for name in ['GroupedMetric', 'threshold', 'confusion', 'ece', 'binary', 'tests']:
        assert ast.dump(next(n for n in old_tree.body if getattr(n, 'name', None) == name)) == ast.dump(next(n for n in new_tree.body if getattr(n, 'name', None) == name))
    frozen.OUT = OUT
    frozen.tests()
    # A mean of per-fit APs is not the AP of a mean probability.
    y = np.array([0, 0, 1, 1])
    p = np.array([[.1, .4, .9, .7], [.9, .8, .7, .6], [.3, .2, .8, .1]])
    mean_ap = np.mean([frozen.average_precision_score(y, s) for s in p])
    pooled_probability_ap = frozen.average_precision_score(y, p.mean(0))
    assert abs(mean_ap - pooled_probability_ap) > 1e-3
    assert np.std([1., 2., 3.], ddof=1) == 1.
    dump('THREE_SEED_STATISTICAL_TESTS.json', {'status': 'PASSED', 'frozen_pure_function_AST_equal': True,
         'sample_SD_ddof': 1, 'mean_of_metrics_not_metric_of_mean_probabilities': True,
         'common_cluster_weights_for_every_seed_and_condition': True})


def merge_tables():
    for table in TABLES:
        pieces = []
        for seed in SEEDS:
            frame = pd.read_csv(OUT / 'seed_statistics' / f's{seed}' / f'{table}.csv')
            frame.insert(0, 'training_seed', seed)
            pieces.append(frame)
        pd.concat(pieces, ignore_index=True).to_csv(OUT / f'ALL_{table}.csv', index=False)


def summaries():
    items = [('METRICS', ['cohort', 'input', 'head'], ['roc_auc','average_precision','brier','binary_nll','ece15']),
             ('INTERNAL_MULTICLASS_METRICS', ['cohort'], ['macro_f1','multiclass_brier_sum7','multiclass_nll','top_label_ece15']),
             ('THRESHOLD_CONFUSION_COUNTS', ['cohort','input','head','target_sensitivity'], ['sensitivity','specificity'])]
    summary, paired = [], []
    for table, tags, metrics in items:
        frame = pd.read_csv(OUT / f'ALL_{table}.csv')
        for key, group in frame.groupby(tags, dropna=False, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            identity = dict(zip(tags, key))
            for metric in metrics:
                matrix = group.pivot(index='training_seed', columns='condition', values=metric).loc[SEEDS, CONDITIONS]
                assert matrix.notna().all().all()
                for condition in CONDITIONS:
                    values = matrix[condition].to_numpy()
                    summary.append({'source_table': table, **identity, 'condition': condition, 'metric': metric,
                         'mean': values.mean(), 'sample_sd': values.std(ddof=1), 'training_seeds': '42;2024;3407', 'n_seeds': 3})
                diffs = (matrix.FULL - matrix.ZERO).to_numpy()
                summary.append({'source_table': table, **identity, 'condition': 'FULL_minus_ZERO', 'metric': metric,
                    'mean': diffs.mean(), 'sample_sd': diffs.std(ddof=1), 'training_seeds': '42;2024;3407', 'n_seeds': 3})
                for seed, value in zip(SEEDS, diffs):
                    paired.append({'source_table': table, **identity, 'training_seed': seed,
                                   'metric': metric, 'difference_FULL_minus_ZERO': value})
    pd.DataFrame(summary).to_csv(OUT / 'THREE_SEED_SUMMARY.csv', index=False)
    pd.DataFrame(paired).to_csv(OUT / 'PER_SEED_PAIRED_DIFFERENCES.csv', index=False)


def bootstrap():
    private_roots = [OLD / 'private_predictions'] + [OUT / 'private_predictions' / f's{s}' for s in SEEDS[1:]]
    alignment = pd.read_csv(private_roots[0] / 'SIIM_alignment.csv', dtype={'patient_id': str, 'image_name': str})
    for root in private_roots[1:]:
        candidate = pd.read_csv(root / 'SIIM_alignment.csv', dtype={'patient_id': str, 'image_name': str})
        pd.testing.assert_frame_equal(alignment, candidate, check_exact=True)
    y = alignment.target.to_numpy()
    patients, cluster = np.unique(alignment.patient_id.astype(str), return_inverse=True)
    assert len(y) == 33126 and len(patients) == 2056 and y.sum() == 584
    calculators, sources = [], []
    for seed, root in zip(SEEDS, private_roots):
        pair = []
        for condition in CONDITIONS:
            path = root / f'{condition}_SIIM_probabilities.npy'
            expected = json.loads((root / f'{condition}_SIIM_hashes.json').read_text())['probabilities_sha256']
            assert sha(path) == expected
            p = np.load(path, allow_pickle=False).astype(np.float64)
            assert p.shape == (33126, 2, 8) and np.isfinite(p).all()
            pair.append([frozen.GroupedMetric(y, p[:, c, s]) for mode, head, c, s in frozen.UNITS])
            sources.append({'training_seed': seed, 'condition': condition, 'prediction_path': str(path), 'sha256': expected})
        calculators.append(pair)
    def evaluate(weights):
        return np.array([[[metric(weights) for metric in condition] for condition in pair] for pair in calculators])
    point = evaluate(np.ones(len(y)))  # seed x condition x input/head x AUC/AP
    metric_table = pd.read_csv(OUT / 'ALL_METRICS.csv')
    for i, seed in enumerate(SEEDS):
        for j, condition in enumerate(CONDITIONS):
            for u, (mode, head, _, _) in enumerate(frozen.UNITS):
                row = metric_table[(metric_table.training_seed == seed) & (metric_table.condition == condition) &
                    (metric_table.cohort == 'SIIM') & (metric_table.input == mode) & (metric_table['head'] == head)].iloc[0]
                assert np.allclose(point[i, j, u], row[['roc_auc', 'average_precision']].to_numpy(dtype=float), atol=1e-12)
    dump('BOOTSTRAP_INPUT_IDENTITY.json', {'sources': sources, 'patient_order': 'np.unique(patient_id.astype(str)), ascending lexical; identical to seed42',
         'patient_order_sha256': hashlib.sha256('\n'.join(patients).encode()).hexdigest(), 'clusters': len(patients), 'images': len(y)})
    rng = np.random.Generator(np.random.PCG64(20261015))
    draws = np.full((10000, 3, 2, 4, 2), np.nan, dtype=np.float64)
    accepted, rejected = 0, 0
    checkpoint = OUT / 'BOOTSTRAP_PROGRESS.json'
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        assert state['prediction_sources'] == sources and state['training_seeds'] == SEEDS
        accepted, rejected = state['accepted'], state['rejected']
        rng.bit_generator.state = state['rng_state']
        draws = np.load(OUT / 'bootstrap_fixed_fits_private.npy', allow_pickle=False)
        assert draws.shape == (10000, 3, 2, 4, 2)
    while accepted < 10000:
        cluster_counts = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))
        result = evaluate(cluster_counts[cluster].astype(float))
        if not np.isfinite(result).all():
            rejected += 1
            assert rejected < 10000
            continue
        draws[accepted] = result
        accepted += 1
        if accepted % 250 == 0:
            np.save(OUT / 'bootstrap_fixed_fits_private.npy', draws, allow_pickle=False)
            dump('BOOTSTRAP_PROGRESS.json', {'accepted': accepted, 'rejected': rejected, 'rng_state': rng.bit_generator.state,
                'prediction_sources': sources, 'training_seeds': SEEDS, 'target': 10000, 'PCG64_seed': 20261015})
            print(f'THREE_SEED_BOOTSTRAP {accepted}/10000', flush=True)
            if time.time() >= frozen.DEADLINE or (OUT / 'STOP_REQUESTED').exists():
                dump('STATISTICS_STATUS.json', {'status': 'PARTIAL', 'accepted': accepted})
                return False
    # This also verifies the original patient draw order and seed42 inputs were reused exactly.
    old_diffs = np.load(OLD / 'bootstrap_differences_private.npy', allow_pickle=False)
    old_delta = draws[:, 0, 0] - draws[:, 0, 1]
    assert np.allclose(old_diffs, old_delta, atol=1e-12, rtol=0), 'Original draw/order reproduction failed'
    rows = []
    for j, condition in enumerate(CONDITIONS + ['FULL_minus_ZERO']):
        values = draws[:, :, j].mean(axis=1) if j < 2 else (draws[:, :, 0] - draws[:, :, 1]).mean(axis=1)
        estimate = point[:, j].mean(axis=0) if j < 2 else (point[:, 0] - point[:, 1]).mean(axis=0)
        variation = point[:, j].std(axis=0, ddof=1) if j < 2 else (point[:, 0] - point[:, 1]).std(axis=0, ddof=1)
        low, high = np.quantile(values, [.025, .975], axis=0)
        for u, (mode, head, _, _) in enumerate(frozen.UNITS):
            for m, metric in enumerate(['roc_auc', 'average_precision']):
                rows.append({'condition': condition, 'cohort': 'SIIM', 'input': mode, 'head': head, 'metric': metric,
                    'mean': estimate[u, m], 'training_seed_sample_sd': variation[u, m], 'ci_low': low[u, m], 'ci_high': high[u, m],
                    'role': 'PRIMARY' if j == 2 and u == 0 and m == 1 else 'SECONDARY_DESCRIPTIVE',
                    'bootstrap_draws': 10000, 'clusters': 2056, 'training_seeds': '42;2024;3407',
                    'CI_scope': 'patient-cluster paired percentile 95%; conditional on three fixed training pairs; not training-seed uncertainty'})
    pd.DataFrame(rows).to_csv(OUT / 'THREE_SEED_PATIENT_CLUSTER_CI.csv', index=False)
    dump('STATISTICS_STATUS.json', {'status': 'COMPLETE', 'accepted_draws': 10000, 'rejected_draws': rejected,
         'training_seeds': SEEDS, 'seed42_original_bootstrap_draws_reproduced': True,
         'primary': 'mean per-seed SIIM E_REAL softmax MEL AP, FULL minus ZERO',
         'sample_SD_ddof': 1, 'calibration_fitted': False, 'external_threshold_selection': False,
         'uncertainty_scope': 'Patients conditional on three fixed training pairs; training-seed variation reported separately'})
    return True


def main():
    tests()
    assert json.loads((OUT / 'EVALUATION_STATUS.json').read_text())['status'] == 'COMPLETE'
    for seed in SEEDS:
        dest = OUT / 'seed_statistics' / f's{seed}'
        dest.mkdir(parents=True, exist_ok=True)
        if seed == 42:
            for table in TABLES:
                source, target = OLD / f'{table}.csv', dest / f'{table}.csv'
                if not target.exists():
                    shutil.copy2(source, target)
                assert sha(source) == sha(target)
        else:
            frozen.OUT = dest
            frozen.PRIVATE = OUT / 'private_predictions' / f's{seed}'
            frozen.compute_metrics()
        print(f'METRICS_READY seed={seed} reused={seed == 42}', flush=True)
    merge_tables()
    summaries()
    bootstrap()


if __name__ == '__main__':
    import sys
    with threadpool_limits(limits=2):
        tests() if '--tests-only' in sys.argv else main()
