"""Analysis: B1 CPU evaluation of existing OncoDerm predictions, 2026-10-11.
Bootstrap: NumPy PCG64 seed 20261011, 10000 valid paired patient-cluster draws.
Runtime/package versions are saved in ENVIRONMENT.json before new metrics.
Only this new output directory is writable. No historical module is imported.
No training, inference, checkpoint loading, calibration, threshold selection,
benchmark, network, cloud, manuscript edit, or B2/C/D execution is performed.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
from threadpoolctl import threadpool_limits, threadpool_info

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PACKAGE = ROOT / 'research_enhancement_B1_20261011_171400_586992/OncoDerm_Research_Enhancement_Start'
STAGE = PACKAGE / 'Reference_Evidence/Stage_A'
PROTOCOL = PACKAGE / 'protocols/B1_analysis_plan.yaml'
SCORES = ['prob_mel', 'prob_mel_endpoint']
DIRECT = ['B1', 'B2', 'B4', 'B6', 'EfficientNet-B0', 'Swin-Tiny', 'DermaMamba port']
REPEATED = ['B2', 'B4', 'B6']
SEEDS = [42, 2024, 3407]
VALIDATION_SEED = 20261012
TOL = 1e-10
ROUND_TOL = 0.000051


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(name, value):
    p = OUT / name
    p.parent.mkdir(exist_ok=True, parents=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)


def read_json(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def write_csv(name, rows):
    p = OUT / name
    p.parent.mkdir(exist_ok=True, parents=True)
    with p.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def log(message):
    line = f'[{now()}] {message}'
    print(line, flush=True)
    with (OUT / 'EXECUTION.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def identify_model(text):
    for model in ['B1', 'B2', 'B4', 'B6']:
        if re.search(r'\b' + model + r'\b', text):
            return model
    for model in DIRECT[4:]:
        if model in text:
            return model
    raise ValueError('Unknown manuscript model label: ' + text)


def load_inputs():
    """Validate exact navigated sources. A bad prediction blocks only its unit."""
    inventory_path = STAGE / 'PREDICTION_INVENTORY.csv'
    inventory = read_csv(inventory_path)
    originals = {str(Path(r['path'])).casefold(): r for r in read_csv(STAGE / 'SOURCE_MANIFEST.csv')}
    observed, problems, checks, aliases = [], [], [], []

    def observe(p, role, expected=''):
        p = Path(p)
        if not p.is_file():
            raise ValueError('Missing exact source: ' + str(p))
        s = p.stat()
        sha = hash_file(p)
        stage_row = originals.get(str(p).casefold())
        expectations = [v for v in [expected, stage_row['sha256_final'] if stage_row else ''] if v]
        ok = all(sha == e for e in expectations)
        observed.append(dict(path=str(p), role=role, bytes=s.st_size, mtime_ns=s.st_mtime_ns, sha256=sha,
                             expected_sha256='|'.join(sorted(set(expectations))), stage_A_match=ok if expectations else 'not_applicable'))
        if not ok:
            raise ValueError('Hash differs from supplied Stage A record: ' + str(p))
        return sha

    plan = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    assert plan['bootstrap']['seed'] == 20261011 and plan['bootstrap']['n_valid_replicates'] == 10000
    assert plan['metrics']['point_rounding_tolerance_abs'] == ROUND_TOL
    assert plan['inputs']['direct_score_column'] == 'prob_mel'
    for p, role in [(PROTOCOL, 'fixed_protocol'), (inventory_path, 'prediction_navigation'),
                    (STAGE / 'SOURCE_MANIFEST.csv', 'Stage_A_hash_reference'),
                    (STAGE / 'EVIDENCE/AUG_direct_baseline_manifest.json', 'historical_direct_manifest'),
                    (PACKAGE / 'Manuscript_R3_1/OncoDerm_Frontiers_MAIN_R3_1.docx', 'R3.1_main'),
                    (PACKAGE / 'Manuscript_R3_1/OncoDerm_Frontiers_SUPPLEMENTARY_R3_1.docx', 'R3.1_supplement'),
                    (PACKAGE / 'Manuscript_R3_1/README_R3_1_技术修订说明.md', 'R3.1_revision_notes'),
                    (PACKAGE / 'Reference_Evidence/Previous_Fact_Check/FACT_CHECK_REPORT.md', 'prior_navigation_only'),
                    (ROOT / 'OncoDerm_Research_Enhancement_Start.zip', 'original_start_package')]:
        observe(p, role)
    mapping_path = Path(plan['inputs']['patient_mapping'])
    observe(mapping_path, 'patient_mapping_read_only_three_fields')
    mapping = pd.read_csv(mapping_path, usecols=['image_name', 'patient_id', 'target'], dtype={'image_name': str, 'patient_id': str}, float_precision='round_trip')
    assert not mapping.isna().any().any(), 'Mapping has missing values'
    assert mapping.image_name.is_unique, 'Mapping image IDs not unique'
    assert set(mapping.target.unique()) == {0, 1}, 'Mapping labels are not binary'
    mapping = mapping.sort_values('image_name', kind='mergesort').reset_index(drop=True)
    y = mapping.target.to_numpy(dtype=np.int64)
    patients = sorted(mapping.patient_id.unique().tolist())
    patient_lookup = {p: i for i, p in enumerate(patients)}
    patient_index = mapping.patient_id.map(patient_lookup).to_numpy(dtype=np.int64)
    assert len(y) == 33126 and len(patients) == 2056 and int(y.sum()) == 584, 'Cohort counts differ from frozen protocol'
    direct_manifest = json.loads((STAGE / 'EVIDENCE/AUG_direct_baseline_manifest.json').read_text(encoding='utf-8'))
    dm = {r['run_id']: r for r in direct_manifest['runs']}
    units, by_run, rows_seen = {}, {}, set()
    for row in inventory:
        p = Path(row['path'])
        model, seed, split = row['model'], int(row['run_seed']), int(row['split_seed'])
        is_aug = row['manifest_link'] == 'August direct_baseline_manifest.runs'
        role = 'AUG_direct' if is_aug else 'STAB_repeated'
        try:
            assert split == 42
            assert (is_aug and seed == 42 and model in DIRECT) or (not is_aug and model in REPEATED and seed in SEEDS)
            sha = observe(p, role, row['sha256'])
            if is_aug:
                assert row['run_id'] in dm
                assert sha == dm[row['run_id']]['external_prediction_sha256'], 'Historical direct manifest hash mismatch'
            df = pd.read_csv(p, usecols=['image_id', 'label'] + SCORES, dtype={'image_id': str}, float_precision='round_trip')
            assert len(df) == 33126 and df.image_id.is_unique, 'Prediction count/ID uniqueness failure'
            assert set(df.image_id) == set(mapping.image_name), 'Prediction/mapping ID sets differ'
            df = df.set_index('image_id').loc[mapping.image_name]
            assert np.array_equal(df.label.to_numpy(), y), 'Labels disagree after ID join'
            scores = df[SCORES].to_numpy(dtype=np.float64)
            assert np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1)), 'Invalid score range or finiteness'
            key = (model, seed)
            if row['run_id'] in by_run:
                prior = by_run[row['run_id']]
                assert prior['sha256'] == sha and prior['key'] == key, 'Same-run copies disagree; stop affected run'
                aliases.append(dict(run_id=row['run_id'], model=model, seed=seed, retained_path=prior['path'], alias_path=str(p), sha256=sha, action='one_analysis_unit'))
            else:
                assert key not in units, 'Multiple distinct runs map to one model/seed'
                units[key] = dict(run_id=row['run_id'], path=str(p), sha256=sha, scores={c: scores[:, i] for i, c in enumerate(SCORES)}, source_role=role)
                by_run[row['run_id']] = dict(sha256=sha, key=key, path=str(p))
            rows_seen.add((role, model, seed))
            checks.append(dict(path=str(p), model=model, split_seed=split, training_seed=seed, run_id=row['run_id'], rows=len(df), images_unique=True, positives=int(y.sum()), patients=len(patients), labels_equal=True, scores_finite_in_unit_interval=True, status='PASS'))
        except Exception as e:
            problems.append(dict(path=str(p), model=model, training_seed=seed, error=str(e)))
            units.pop((model, seed), None)
    for model in DIRECT:
        if ('AUG_direct', model, 42) not in rows_seen:
            problems.append(dict(model=model, error='Required AUG direct unit not validated'))
    for model in REPEATED:
        for seed in SEEDS:
            if ('STAB_repeated', model, seed) not in rows_seen:
                problems.append(dict(model=model, seed=seed, error='Required STAB repeated unit not validated'))
    # A failure in one copy must not be masked by another copy later in the loop.
    for problem in problems:
        units.pop((problem.get('model'), problem.get('training_seed', problem.get('seed', 42))), None)
    return dict(plan=plan, units=units, y=y, patient_index=patient_index, n_patients=len(patients),
                patient_sort_sha256=hashlib.sha256(('\n'.join(patients) + '\n').encode()).hexdigest(),
                image_sort_sha256=hashlib.sha256(('\n'.join(mapping.image_name) + '\n').encode()).hexdigest(),
                observed=observed, checks=checks, aliases=aliases, problems=problems)


def prepare():
    d = load_inputs()
    write_csv('INPUT_MANIFEST.csv', d['observed'])
    write_json('INPUT_VALIDATION.json', dict(checks=d['checks'], aliases=d['aliases'], problems=d['problems'],
                unique_analysis_runs=len(d['units']), patient_sort_sha256=d['patient_sort_sha256'], image_sort_sha256=d['image_sort_sha256'],
                patient_order='Python lexicographic sort of exact patient_id strings, no numeric conversion',
                image_alignment='Exact image_id -> image_name join then lexicographic image_name order; no row dropping',
                privacy='Only public dataset linkage IDs and target used from official mapping; no clinical fields copied'))
    env = dict(recorded_at=now(), executable=sys.executable, python=sys.version, platform=platform.platform(),
               packages={m.__name__: m.__version__ for m in [np, pd, scipy, sklearn, yaml]},
               workers=1, blas_threads=1, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], threadpools=threadpool_info())
    write_json('ENVIRONMENT.json', env)
    frozen = dict(d['plan'])
    frozen.update(status='frozen_before_new_performance_computation', frozen_at=now(), original_protocol_sha256=hash_file(PROTOCOL),
        input_manifest_sha256=hash_file(OUT / 'INPUT_MANIFEST.csv'), independent_script_sha256=hash_file(Path(__file__)),
        environment_sha256=hash_file(OUT / 'ENVIRONMENT.json'),
        interpretation='Supplemental analysis of previously inspected SIIM cohort; not July historical preregistration',
        implementation=dict(dtype='float64', csv_float_precision='round_trip', numpy_quantile_method='linear',
            patient_order='lexicographic exact patient_id strings', image_order='lexicographic image_name via exact ID join',
            workers=1, blas_threads=1, pilot_replicates=100, pilot_included_in_total=True,
            validation_rng_seed=VALIDATION_SEED, validation_real_draws=3, validation_stream_independent_of_bootstrap=True,
            resume_snapshots='local_only/checkpoint_*.npz and same-number RNG JSON; no patients/predictions saved',
            source_hash_guard='Before analysis and final read-only check', plots='coordinates_only_no_image_rendering'))
    with (OUT / 'B1_FROZEN_PLAN.yaml').open('x', encoding='utf-8') as f:
        yaml.safe_dump(frozen, f, allow_unicode=True, sort_keys=False)
    write_json('FREEZE_HASHES.json', {n: hash_file(OUT / n) for n in ['B1_FROZEN_PLAN.yaml', 'INPUT_MANIFEST.csv', 'ENVIRONMENT.json', 'b1_analysis.py']})
    log(f'Frozen before metric computation; sources={len(d["observed"])} validated_csv={len(d["checks"])} unique_runs={len(d["units"])} aliases={len(d["aliases"])} problems={len(d["problems"])}')


class GroupedMetric:
    """Tie-grouped descending-score weighted AUC/AP with no score clipping."""
    def __init__(self, y, score):
        self.order = np.argsort(-score, kind='mergesort')
        s = score[self.order]
        self.starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
        self.y = y[self.order].astype(np.float64)

    def __call__(self, weights):
        w = weights[self.order]
        positive = np.add.reduceat(w * self.y, self.starts)
        negative = np.add.reduceat(w * (1.0 - self.y), self.starts)
        p, n = positive.sum(), negative.sum()
        if p <= 0 or n <= 0:
            return np.array([np.nan, np.nan])
        tp = np.cumsum(positive)
        fp = np.cumsum(negative)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        ap = np.sum(positive * precision) / p
        auc = np.sum(negative * (tp - 0.5 * positive)) / (p * n)
        return np.array([auc, ap], dtype=np.float64)


def reference(y, score, weight=None):
    return np.array([roc_auc_score(y, score, sample_weight=weight), average_precision_score(y, score, sample_weight=weight)])


def implementation_tests():
    rng = np.random.Generator(np.random.PCG64(17011))
    cases = [
        ('distinct_scores', [0, 1, 0, 1, 0, 1], [.1, .9, .3, .5, .2, .6], [1]*6),
        ('tied_scores', [0, 1, 0, 1, 0, 1], [.1, .5, .5, .5, .9, .9], [1]*6),
        ('constant_scores', [0, 1, 0, 1], [.4]*4, [1, 1, 1, 1]),
        ('nonuniform_integer_weights', [0, 1, 0, 1, 1], [.7, .9, .3, .7, .1], [3, 1, 5, 2, 4]),
        ('zero_weight_score_groups', [0, 1, 0, 1, 0, 1], [.9, .8, .7, .7, .3, .1], [0, 0, 3, 2, 0, 1]),
        ('minimum_two_classes', [0, 1], [.2, .8], [2, 3]),
    ]
    patient_indices = np.array([0, 0, 1, 2, 2, 3, 3, 3])
    multiplicity = np.bincount(np.array([0, 0, 2, 3]), minlength=4)
    cases.append(('patient_repeat_weights', [0, 1, 0, 1, 0, 0, 1, 1], [.1, .8, .4, .8, .7, .2, .5, .5], multiplicity[patient_indices]))
    for i in range(12):
        yy = np.tile([0, 1], 10)
        ss = rng.random(20) if i % 2 else rng.integers(0, 5, 20) / 4
        ww = rng.integers(0, 5, 20)
        ww[:2] += 1
        cases.append((f'random_{i:02}', yy, ss, ww))
    records = []
    for name, yy, ss, ww in cases:
        y, score, weight = np.array(yy), np.array(ss, dtype=float), np.array(ww, dtype=float)
        got = GroupedMetric(y, score)(weight)
        ref = reference(y, score, weight)
        repeat = np.repeat(np.arange(len(y)), weight.astype(int))
        expanded = reference(y[repeat], score[repeat])
        err = max(np.max(np.abs(got-ref)), np.max(np.abs(got-expanded)))
        assert err <= TOL, (name, got, ref, expanded)
        records.append(dict(case=name, grouped=got, sklearn_weighted=ref, sklearn_expanded=expanded, max_abs_error=err, passed=True))
    for name, y, w in [('single_positive', np.ones(4), np.ones(4)), ('single_negative', np.zeros(4), np.ones(4)), ('effective_single_class', np.array([0, 1, 1, 0]), np.array([0, 1, 3, 0])), ('zero_total_weight', np.array([0, 1, 0, 1]), np.zeros(4))]:
        assert np.isnan(GroupedMetric(y, np.array([.1, .2, .3, .4]))(w)).all()
        records.append(dict(case=name, grouped='undefined_two_class_metric', passed=True))
    return records


def manuscript_checks(d, points, repeated_rows, summaries):
    tables = read_json('R3_1_TABLES_BY_CAPTION.json')
    checks = []
    def add(table, model, seed, score, statistic, metric, value, display):
        old = float(display)
        err = abs(value - old)
        checks.append(dict(table=table, model=model, training_seed=seed, score=score, statistic=statistic, metric=metric,
                           recomputed=value, manuscript_display=display, absolute_difference=err, tolerance=ROUND_TOL, passed=err <= ROUND_TOL))
    for row in tables['MAIN_1']['rows'][1:]:
        model = identify_model(row[0])
        if model not in points:
            continue
        for metric, index in [('roc_auc', 3), ('average_precision', 4)]:
            add('R3.1_Table1', model, 42, 'prob_mel', 'point', metric, points[model][metric], row[index])
    repeated_lookup = {(r['model'], r['training_seed'], r['score']): r for r in repeated_rows}
    for row in tables['SUPPLEMENTARY_S13']['rows'][1:]:
        model, seed = row[0], int(row[2])
        found = repeated_lookup.get((model, seed, 'prob_mel_endpoint'))
        if found:
            for metric, index in [('roc_auc', 6), ('average_precision', 7)]:
                add('R3.1_TableS13', model, seed, 'prob_mel_endpoint', 'point', metric, found[metric], row[index])
    for row in tables['MAIN_3']['rows'][1:]:
        model = identify_model(row[0])
        for metric, index in [('roc_auc', 3), ('average_precision', 4)]:
            found = [s for s in summaries if s['model'] == model and s['score'] == 'prob_mel_endpoint' and s['metric'] == metric]
            if not found:
                continue
            mean, sd = re.fullmatch(r'([\d.]+)\s*\(([\d.]+)\)', row[index]).groups()
            for statistic, value, display in [('mean', found[0]['mean'], mean), ('sample_sd_ddof1', found[0]['sample_sd_ddof1'], sd)]:
                add('R3.1_Table3', model, '42|2024|3407', 'prob_mel_endpoint', statistic, metric, value, display)
    if 'B6' in points and 'DermaMamba port' in points:
        for row in tables['MAIN_2']['rows'][1:]:
            metric = 'roc_auc' if row[0] == 'ROC-AUC' else 'average_precision'
            add('R3.1_Table2', 'B6_minus_DermaMamba port', 42, 'prob_mel', 'difference', metric,
                points['B6'][metric] - points['DermaMamba port'][metric], row[3])
    write_csv('MANUSCRIPT_POINT_CHECKS.csv', checks)
    return checks


def compute_points(d):
    points, repeated_rows, summaries = {}, [], []
    for model in DIRECT:
        unit = d['units'].get((model, 42))
        if unit is None:
            continue
        auc, ap = reference(d['y'], unit['scores']['prob_mel'])
        points[model] = dict(model=model, split_seed=42, training_seed=42, score='prob_mel', run_id=unit['run_id'],
                             images=len(d['y']), patients=d['n_patients'], positive_images=int(d['y'].sum()),
                             prevalence=float(d['y'].mean()), roc_auc=auc, average_precision=ap, input_sha256=unit['sha256'])
    for model in REPEATED:
        for seed in SEEDS:
            unit = d['units'].get((model, seed))
            if unit is None:
                continue
            for score in SCORES:
                auc, ap = reference(d['y'], unit['scores'][score])
                repeated_rows.append(dict(model=model, split_seed=42, training_seed=seed, score=score, run_id=unit['run_id'],
                                         images=len(d['y']), roc_auc=auc, average_precision=ap, input_sha256=unit['sha256']))
        for score in SCORES:
            selected = [r for r in repeated_rows if r['model'] == model and r['score'] == score]
            if len(selected) == 3:
                for metric in ['roc_auc', 'average_precision']:
                    values = np.array([r[metric] for r in selected])
                    summaries.append(dict(model=model, split_seed=42, score=score, metric=metric, training_seeds='42|2024|3407',
                        n_unique_runs=3, mean=values.mean(), sample_sd_ddof1=values.std(ddof=1), minimum=values.min(), maximum=values.max(),
                        provenance='B1_new_common_score_analysis' if score == 'prob_mel' else 'B1_recomputed_dedicated_head_with_R3.1_check'))
    write_csv('DIRECT_POINT_METRICS.csv', list(points.values()))
    write_csv('REPEATED_RUN_METRICS.csv', repeated_rows)
    write_csv('REPEATED_RUN_SUMMARY.csv', summaries)
    checks = manuscript_checks(d, points, repeated_rows, summaries)
    log(f'Point estimates computed; R3.1 checks passed {sum(r["passed"] for r in checks)}/{len(checks)}')
    return points, repeated_rows, summaries, checks


def generate_curves(d, points):
    tests = []
    with (OUT / 'ROC_CURVE_COORDINATES.csv').open('x', encoding='utf-8', newline='') as rf, (OUT / 'PR_CURVE_COORDINATES.csv').open('x', encoding='utf-8', newline='') as pf:
        rw = csv.DictWriter(rf, fieldnames=['model', 'split_seed', 'training_seed', 'score', 'point_index', 'false_positive_rate', 'true_positive_rate', 'threshold'])
        pw = csv.DictWriter(pf, fieldnames=['model', 'split_seed', 'training_seed', 'score', 'point_index', 'recall', 'precision', 'threshold'])
        rw.writeheader()
        pw.writeheader()
        for model in DIRECT:
            if model not in points:
                continue
            score = d['units'][(model, 42)]['scores']['prob_mel']
            fpr, tpr, rt = roc_curve(d['y'], score, drop_intermediate=False)
            precision, recall, pt = precision_recall_curve(d['y'], score, drop_intermediate=False)
            assert (fpr[0], tpr[0]) == (0, 0) and (fpr[-1], tpr[-1]) == (1, 1)
            assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
            assert recall[0] == 1 and recall[-1] == 0 and precision[-1] == 1
            assert abs(precision[0] - d['y'].mean()) <= TOL and np.all(np.diff(recall) <= 0)
            curve_ap = -np.sum(np.diff(recall) * precision[:-1])
            assert abs(curve_ap - points[model]['average_precision']) <= TOL
            assert abs(np.trapezoid(tpr, fpr) - points[model]['roc_auc']) <= TOL
            base = dict(model=model, split_seed=42, training_seed=42, score='prob_mel')
            for i in range(len(fpr)):
                rw.writerow(dict(base, point_index=i, false_positive_rate=fpr[i], true_positive_rate=tpr[i], threshold=rt[i]))
            for i in range(len(recall)):
                pw.writerow(dict(base, point_index=i, recall=recall[i], precision=precision[i], threshold=pt[i] if i < len(pt) else ''))
            tests.append(dict(model=model, roc_points=len(fpr), pr_points=len(recall), PR_x='recall', PR_y='precision',
                             pr_endpoint_first=[recall[0], precision[0]], pr_endpoint_last=[recall[-1], precision[-1]],
                             ap_from_recall_increments=curve_ap, ap_absolute_error=abs(curve_ap - points[model]['average_precision']), passed=True))
    write_json('CURVE_TESTS.json', tests)


def bootstrap_design(d, checks):
    bad_direct = {r['model'] for r in checks if r['table'] == 'R3.1_Table1' and not r['passed']}
    bad_heads = {(r['model'], r['score']) for r in checks if r['table'] in ['R3.1_Table3', 'R3.1_TableS13'] and not r['passed']}
    keys = [(m, 42, 'prob_mel') for m in DIRECT if (m, 42) in d['units'] and m not in bad_direct]
    comparisons = []
    blocked = []
    for c in d['plan']['comparisons']:
        if all((m, 42, 'prob_mel') in keys for m in [c['left'], c['right']]):
            comparisons.append(c)
        else:
            blocked.append(dict(comparison=c, reason='Missing/invalid input or direct-score manuscript mismatch'))
    fixed_heads = []
    for score in SCORES:
        good = all((m, seed) in d['units'] for m in ['B2', 'B6'] for seed in SEEDS)
        good = good and not any((m, score) in bad_heads for m in ['B2', 'B6'])
        if score == 'prob_mel':
            good = good and not (bad_direct & {'B2', 'B6'})
        if good:
            fixed_heads.append(score)
            for model in ['B2', 'B6']:
                for seed in SEEDS:
                    key = (model, seed, score)
                    if key not in keys:
                        keys.append(key)
        else:
            blocked.append(dict(fixed_three_runs_score=score, reason='Missing/invalid unit or score-path mismatch'))
    return keys, comparisons, fixed_heads, blocked


def real_weight_tests(d, keys, metrics):
    rng = np.random.Generator(np.random.PCG64(VALIDATION_SEED))
    records = []
    for draw in range(3):
        multiplicities = np.bincount(rng.integers(0, d['n_patients'], size=d['n_patients']), minlength=d['n_patients'])
        weights = multiplicities[d['patient_index']].astype(np.float64)
        for key, metric in zip(keys, metrics):
            ref = reference(d['y'], d['units'][key[:2]]['scores'][key[2]], weights)
            got = metric(weights)
            error = float(np.max(np.abs(got - ref)))
            assert error <= TOL, (key, draw, error)
            records.append(dict(draw=draw, model=key[0], training_seed=key[1], score=key[2], grouped=got, sklearn_weighted=ref, max_abs_error=error, passed=True))
    return records


def snapshot(count, array, keys, rng, attempts, rejected, elapsed, checkpoint_number):
    local = OUT / 'local_only'
    local.mkdir(exist_ok=True)
    stem = f'checkpoint_{checkpoint_number:06}_{count:05}'
    with (local / (stem + '.npz')).open('xb') as f:
        np.savez_compressed(f, metrics=array[:count])
    write_json('local_only/' + stem + '.json', dict(completed_valid=count, attempts=attempts, rejected_single_class=rejected,
        elapsed_seconds=elapsed, keys=keys, rng_state=rng.bit_generator.state,
        protocol_sha256=hash_file(OUT / 'B1_FROZEN_PLAN.yaml'), input_manifest_sha256=hash_file(OUT / 'INPUT_MANIFEST.csv'),
        array_sha256=hash_file(local / (stem + '.npz')), metrics_order=['roc_auc', 'average_precision']))
    return local / (stem + '.json')


def run_bootstrap(d, keys, metrics, resume=False):
    n = d['plan']['bootstrap']['n_valid_replicates']
    rng = np.random.Generator(np.random.PCG64(20261011))
    values = np.empty((n, len(keys), 2), dtype=np.float64)
    count = attempts = rejected = checkpoint_number = 0
    elapsed_prior = 0.0
    if resume:
        state_path = sorted((OUT / 'local_only').glob('checkpoint_*.json'))[-1]
        state = json.loads(state_path.read_text(encoding='utf-8'))
        assert state['protocol_sha256'] == hash_file(OUT / 'B1_FROZEN_PLAN.yaml')
        assert state['input_manifest_sha256'] == hash_file(OUT / 'INPUT_MANIFEST.csv')
        assert state['keys'] == [list(k) for k in keys]
        npz = state_path.with_suffix('.npz')
        assert hash_file(npz) == state['array_sha256']
        with np.load(npz, allow_pickle=False) as saved:
            count = state['completed_valid']
            values[:count] = saved['metrics']
        rng.bit_generator.state = state['rng_state']
        attempts, rejected = state['attempts'], state['rejected_single_class']
        elapsed_prior = state['elapsed_seconds']
        checkpoint_number = int(state_path.name.split('_')[1])
        log(f'Resuming the same RNG stream at valid replicate {count}; no seed changes')
    start = time.perf_counter()
    pilot = None
    while count < n:
        attempts += 1
        multiplicity = np.bincount(rng.integers(0, d['n_patients'], size=d['n_patients']), minlength=d['n_patients'])
        weights = multiplicity[d['patient_index']].astype(np.float64)
        positive_weight = weights @ d['y']
        if positive_weight <= 0 or weights.sum() - positive_weight <= 0:
            rejected += 1
            continue
        for j, metric in enumerate(metrics):
            values[count, j] = metric(weights)
        assert np.isfinite(values[count]).all()
        count += 1
        if count == 100 or count % 1000 == 0:
            elapsed = elapsed_prior + time.perf_counter() - start
            checkpoint_number += 1
            state_path = snapshot(count, values, keys, rng, attempts, rejected, elapsed, checkpoint_number)
            if count == 100:
                pilot = dict(valid_replicates=100, elapsed_seconds=elapsed, projected_total_seconds=elapsed / count * n,
                    same_fixed_stream=True, included_in_final_10000=True, numeric_tests_passed=True, state_path=str(state_path))
                write_json('PILOT_100.json', pilot)
                log(f'Pilot100 passed; projected total bootstrap {pilot["projected_total_seconds"]:.1f}s; continuing same stream')
                if pilot['projected_total_seconds'] > 4 * 3600:
                    return values[:count], dict(status='PARTIAL_TIME_LIMIT_REVIEW', completed_valid=count, attempts=attempts, rejected_single_class=rejected, elapsed_seconds=elapsed, resume_command=f'"{sys.executable}" -B "{__file__}" --resume')
            log(f'Bootstrap {count}/{n} valid; attempts={attempts}; rejected={rejected}; elapsed={elapsed:.1f}s')
            if time.perf_counter() - start > 4 * 3600 and count < n:
                return values[:count], dict(status='PARTIAL_TIME_LIMIT', completed_valid=count, attempts=attempts, rejected_single_class=rejected, elapsed_seconds=elapsed, resume_command=f'"{sys.executable}" -B "{__file__}" --resume')
    return values, dict(status='COMPLETE', completed_valid=count, attempts=attempts, rejected_single_class=rejected,
                        elapsed_seconds=elapsed_prior + time.perf_counter() - start, generator='PCG64', seed=20261011,
                        resample_shared_across_all_score_vectors=True, score_vectors=len(keys))


def summarize_bootstrap(d, values, keys, comparisons, fixed_heads, points, repeated_rows):
    indexes = {key: i for i, key in enumerate(keys)}
    metric_names = ['roc_auc', 'average_precision']
    pairs = []
    for c in comparisons:
        left, right = c['left'], c['right']
        for j, metric in enumerate(metric_names):
            delta = values[:, indexes[(left, 42, 'prob_mel')], j] - values[:, indexes[(right, 42, 'prob_mel')], j]
            lo, hi = np.quantile(delta, [.025, .975], method='linear')
            family = metric == 'average_precision' and c['role'].startswith('primary_AP')
            family_ci = np.quantile(delta, [.0125, .9875], method='linear') if family else ['', '']
            pairs.append(dict(left=left, right=right, role=c['role'], score='prob_mel', training_seed=42, split_seed=42,
                metric=metric, left_point=points[left][metric], right_point=points[right][metric], difference=points[left][metric]-points[right][metric],
                nominal_95_lower=lo, nominal_95_upper=hi, primary_AP_97_5_lower=family_ci[0], primary_AP_97_5_upper=family_ci[1],
                family_adjustment='approximate_Bonferroni_two_primary_AP_comparisons' if family else 'not_applied',
                valid_replicates=len(values), resampling_unit='patient', estimand='image_weighted_metric_conditional_on_fixed_runs', p_value='not_computed'))
    if pairs:
        write_csv('PAIRWISE_CLUSTERED_CI.csv', pairs)
    rep = {(r['model'], r['training_seed'], r['score']): r for r in repeated_rows}
    fixed = []
    for score in fixed_heads:
        for j, metric in enumerate(metric_names):
            deltas = np.stack([values[:, indexes[('B6', s, score)], j]-values[:, indexes[('B2', s, score)], j] for s in SEEDS], axis=1)
            point_differences = [rep[('B6', s, score)][metric]-rep[('B2', s, score)][metric] for s in SEEDS]
            lo, hi = np.quantile(deltas.mean(axis=1), [.025, .975], method='linear')
            fixed.append(dict(left='B6', right='B2', score=score, metric=metric, split_seed=42, training_seeds='42|2024|3407',
                mean_difference=float(np.mean(point_differences)), nominal_95_lower=lo, nominal_95_upper=hi,
                seed42_difference=point_differences[0], seed2024_difference=point_differences[1], seed3407_difference=point_differences[2],
                valid_replicates=len(values), interpretation='conditional_on_fixed_three_runs', seed_population_inference=False, p_value='not_computed'))
    if fixed:
        write_csv('FIXED_RUN_MEAN_DIFFERENCE_CI.csv', fixed)
    return pairs, fixed


def verify_frozen_inputs():
    frozen = read_json('FREEZE_HASHES.json')
    for name, sha in frozen.items():
        assert hash_file(OUT / name) == sha, 'Frozen file changed: ' + name
    for row in read_csv(OUT / 'INPUT_MANIFEST.csv'):
        assert hash_file(row['path']) == row['sha256'], 'Input changed after freeze: ' + row['path']


def compute(resume=False):
    verify_frozen_inputs()
    d = load_inputs()
    if not resume:
        toy_tests = implementation_tests()
        write_json('SYNTHETIC_METRIC_TESTS.json', toy_tests)
        log(f'Grouped metric implementation passed {len(toy_tests)} synthetic cases before real-data metrics')
        points, repeated_rows, summaries, checks = compute_points(d)
        generate_curves(d, points)
    else:
        points = {r['model']: r for r in read_csv(OUT / 'DIRECT_POINT_METRICS.csv')}
        repeated_rows = read_csv(OUT / 'REPEATED_RUN_METRICS.csv')
        summaries = read_csv(OUT / 'REPEATED_RUN_SUMMARY.csv')
        checks = read_csv(OUT / 'MANUSCRIPT_POINT_CHECKS.csv')
        for r in list(points.values()) + repeated_rows:
            for col in ['roc_auc', 'average_precision']:
                r[col] = float(r[col])
            r['training_seed'] = int(r['training_seed'])
        for r in checks:
            r['passed'] = r['passed'] == 'True'
    keys, comparisons, fixed_heads, blocked = bootstrap_design(d, checks)
    metrics = [GroupedMetric(d['y'], d['units'][k[:2]]['scores'][k[2]]) for k in keys]
    if not resume:
        real_tests = real_weight_tests(d, keys, metrics)
        write_json('METRIC_TESTS.json', dict(tolerance=TOL, synthetic_cases=read_json('SYNTHETIC_METRIC_TESTS.json'),
            real_weight_checks=real_tests, validation_seed=VALIDATION_SEED, validation_stream_is_separate=True, all_passed=True,
            maximum_real_abs_error=max((r['max_abs_error'] for r in real_tests), default=0)))
        write_json('BOOTSTRAP_DESIGN.json', dict(keys=keys, comparisons=comparisons, fixed_heads=fixed_heads, blocked=blocked))
        log(f'Real-weight sklearn cross-checks passed: {len(real_tests)}; starting shared bootstrap for {len(keys)} score vectors')
    if not comparisons and not fixed_heads:
        write_json('B1_STATUS.json', dict(status='BLOCKED', reason='No unblocked planned bootstrap analyses', blocked=blocked, input_problems=d['problems']))
        return
    values, status = run_bootstrap(d, keys, metrics, resume)
    if status['status'] == 'COMPLETE':
        pairs, fixed = summarize_bootstrap(d, values, keys, comparisons, fixed_heads, points, repeated_rows)
        status.update(pairwise_rows=len(pairs), fixed_run_mean_rows=len(fixed), blocked=blocked, input_problems=d['problems'],
                      overall_status='COMPLETE' if not blocked and not d['problems'] and all(r['passed'] for r in checks) else 'PARTIAL')
    write_json('B1_STATUS.json' if not (OUT / 'B1_STATUS.json').exists() else 'B1_STATUS_RESUMED.json', status)
    log(json.dumps(status, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--compute', action='store_true')
    group.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.prepare:
            prepare()
        else:
            compute(args.resume)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        log('STOPPED: ' + type(exc).__name__ + ': ' + str(exc))
        raise
