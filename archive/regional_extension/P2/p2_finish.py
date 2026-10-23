"""Frozen P1 metric definitions; P2 validation-only summaries and delivery."""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import difflib
import hashlib
import json
import math
import statistics
import time
import zipfile

OUT = Path(__file__).resolve().parent
PKG = OUT / 'FIM_RegionClinical_Dev_Start'
SEEDS = [2024, 3407]
VARIANTS = ['R_IMG', 'R_CLIN', 'XATTN']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def table(path, rows):
    if rows:
        with Path(path).open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)


def evaluate_run(training_seed, variant):
    import gc
    import numpy as np
    import torch
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
    import p2_training as t
    assert training_seed in SEEDS and variant in VARIANTS
    root = OUT / f'{variant}_s{training_seed}'
    status_path = root / 'status.json'
    if not status_path.exists() or read(status_path)['status'] != 'TRAINING_COMPLETE':
        return False
    if (root / 'VALIDATION_RESULT.json').exists():
        assert read(root / 'VALIDATION_RESULT.json')['best_sha256'] == sha(root / 'best.pt')
        return True
    if time.time() >= t.DEADLINE:
        t.log('PARTIAL: deadline before selected-best validation summary')
        return False
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert torch.cuda.is_available()
    val = t.HAMDataset(t.config(training_seed), 'val')
    assert len(val) == 1513
    t.seed_all(training_seed)
    state = torch.load(root / 'best.pt', map_location='cpu', weights_only=False)
    cfg = state['config']['original_config']
    assert cfg['training_seed'] == training_seed and cfg['p1_variant'] == variant
    model = t.fresh_model(cfg)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    expected, best_epoch = state['selection_validation'], state['epoch']
    assert best_epoch == read(status_path)['best_epoch']
    del state
    probabilities, endpoints, targets = [], [], []
    diagnostics = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for images, metadata, labels, _ in t.loader(val, list(range(len(val))), batch_size=8):
            if time.time() >= t.DEADLINE:
                t.log('PARTIAL: deadline during validation; no incomplete metrics exported')
                return False
            output = model(images.cuda(non_blocking=True), metadata.cuda(non_blocking=True))
            probabilities.append(output['logits'].softmax(1).cpu().numpy())
            endpoints.append(output['mel_logits'].sigmoid().cpu().numpy())
            targets.append(labels.numpy())
            t.merge_diagnostics(diagnostics, t.readout_diagnostics(output, 'validation'))
    prob = np.concatenate(probabilities).astype(np.float64)
    endpoint = np.concatenate(endpoints).astype(np.float64)
    y7 = np.concatenate(targets)
    y = (y7 == 4).astype(int)
    assert np.isfinite(prob).all() and np.isfinite(endpoint).all()
    assert np.allclose(prob.sum(1), 1, atol=1e-6)
    metrics = {'macro_f1': float(f1_score(y7, prob.argmax(1), labels=list(range(7)), average='macro')),
        'multiclass_brier_sum7': float(np.mean(np.sum((prob - np.eye(7)[y7])**2, axis=1))),
        'multiclass_nll': float(np.mean(-np.log(np.clip(prob[np.arange(len(y7)), y7], 1e-7, 1))))}
    binary = []
    for head, score in [('softmax_MEL', prob[:, 4]), ('endpoint_MEL', endpoint)]:
        clipped = np.clip(score, 1e-7, 1 - 1e-7)
        binary.append({'head': head, 'roc_auc': float(roc_auc_score(y, score)),
            'average_precision': float(average_precision_score(y, score)),
            'brier': float(np.mean((score - y)**2)),
            'binary_nll': float(np.mean(-y * np.log(clipped) - (1-y) * np.log1p(-clipped))),
            'images': len(y), 'positive_images': int(y.sum())})
    for actual, original in [(metrics['macro_f1'], expected['macro_f1']),
        (binary[0]['roc_auc'], expected['mel_auc']), (binary[0]['average_precision'], expected['mel_ap'])]:
        assert abs(actual - original) < 1e-7, 'Selected-best validation mismatch'
    result = {'status': 'COMPLETE', 'run': f'{variant}_s{training_seed}', 'training_seed': training_seed,
        'cohort': 'HAM_val', 'input': 'NATIVE', 'best_path': str(root / 'best.pt'),
        'best_sha256': sha(root / 'best.pt'), 'selected_epoch': best_epoch,
        'dtype': 'FP32', 'TF32': False, 'strict_loading': True, 'multiclass': metrics, 'binary': binary,
        'selection_metrics_reproduced': True, 'readout_diagnostics': t.summarize_diagnostics(diagnostics),
        'eta_raw': float(model.readout.eta.detach()), 'evaluation_seconds': time.perf_counter() - started,
        'per_example_predictions_exported': False}
    dump(root / 'VALIDATION_RESULT.json', result)
    t.log(f'VALIDATION_SUMMARY {variant}_s{training_seed} selected_epoch={best_epoch} AP={binary[0]["average_precision"]:.6f}')
    del model, probabilities, endpoints, targets, prob, endpoint
    gc.collect()
    torch.cuda.empty_cache()
    return True


MULTI_KEYS = ['macro_f1', 'multiclass_brier_sum7', 'multiclass_nll']
BINARY_KEYS = ['roc_auc', 'average_precision', 'brier', 'binary_nll']


def metric_rows(model, seed, source, multi, binary):
    rows = [{'model': model, 'training_seed': seed, 'origin': source, 'head': 'seven_class',
        'metric': k, 'value': float(multi[k])} for k in MULTI_KEYS]
    for b in binary:
        rows.extend({'model': model, 'training_seed': seed, 'origin': source, 'head': b['head'],
            'metric': k, 'value': float(b[k])} for k in BINARY_KEYS)
    assert all(math.isfinite(r['value']) for r in rows)
    return rows


def summarize(rows, group_keys):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)
    output = []
    for key, group in sorted(groups.items()):
        seeds = sorted(r['training_seed'] for r in group)
        assert len(seeds) == len(set(seeds))
        values = [r['value'] for r in group]
        output.append({**dict(zip(group_keys, key)), 'n_seeds': len(values), 'seeds': ','.join(map(str, seeds)),
            'three_seed_complete': seeds == [42, 2024, 3407], 'mean': statistics.mean(values),
            'sample_SD_ddof1': statistics.stdev(values) if len(values) > 1 else None})
    return output


def package():
    from p2_prepare import verify_freeze
    assert not (OUT / 'P2_TRAIN_RESULTS.zip').exists(), 'Do not overwrite an existing delivery'
    verify_freeze()
    window = read(OUT / 'WINDOW.json')
    errors = sorted(OUT.glob('*ERROR_*.json'))
    identities, metrics, diagnostics = [], [], []
    completed = 0
    for seed in [42, *SEEDS]:
        for variant in VARIANTS:
            root = (OUT / 'reused_P1' if seed == 42 else OUT) / f'{variant}_s{seed}'
            origin = 'reused_P1_no_execution' if seed == 42 else 'new_P2_training'
            status = read(root / 'status.json') if (root / 'status.json').exists() else {'status': 'NOT_STARTED', 'completed_epochs': 0}
            row = {'run': root.name, 'model': variant, 'training_seed': seed, 'origin': origin,
                'training_status': status['status'], 'completed_epochs': status['completed_epochs'],
                'selected_epoch': status.get('best_epoch'), 'validation_complete': False}
            if (root / 'effective_config.json').exists():
                cfg = read(root / 'effective_config.json')
                row.update(parameter_count=cfg['parameter_count'], new_readout_parameter_count=cfg['readout_parameter_count'])
            if (root / 'training_history.csv').exists():
                with (root / 'training_history.csv').open(encoding='utf-8', newline='') as f:
                    hist = list(csv.DictReader(f))
                row['sum_completed_epoch_seconds'] = sum(float(h['epoch_seconds']) for h in hist)
                row['run_wall_seconds_last_invocation'] = status.get('elapsed_this_invocation_seconds')
            if (root / 'VALIDATION_RESULT.json').exists():
                result = read(root / 'VALIDATION_RESULT.json')
                assert result['cohort'] == 'HAM_val' and result['input'] == 'NATIVE'
                if seed != 42:
                    assert sha(root / 'best.pt') == result['best_sha256']
                    completed += int(status['status'] == 'TRAINING_COMPLETE')
                row.update(validation_complete=True, best_sha256=result['best_sha256'], **result['multiclass'])
                row.update(softmax_MEL_AP=result['binary'][0]['average_precision'], softmax_MEL_AUC=result['binary'][0]['roc_auc'])
                metrics.extend(metric_rows(variant, seed, origin, result['multiclass'], result['binary']))
                diagnostics.append({'run': root.name, 'origin': origin, 'selected_epoch': result['selected_epoch'],
                    'eta_raw': result['eta_raw'], 'regional_measures_applicable': variant != 'XATTN', **result['readout_diagnostics']})
            identities.append(row)
    for ref in read(OUT / 'M0_VALIDATION_REFERENCES.json')['references']:
        seed, identity = ref['training_seed'], ref['identity']
        metrics.extend(metric_rows('M0', seed, 'readonly_current_FIM_FULL', ref['multiclass'], ref['binary']))
        identities.append({'run': f'FIM_LPAH_FULL_s{seed}', 'model': 'M0', 'training_seed': seed,
            'origin': 'readonly_current_FIM_FULL', 'training_status': ref['status']['status'],
            'completed_epochs': ref['status']['completed_epochs'], 'selected_epoch': identity['best_epoch'],
            'best_sha256': identity['sha256'], 'validation_complete': True,
            **{k: float(ref['multiclass'][k]) for k in MULTI_KEYS},
            'softmax_MEL_AP': float(next(b['average_precision'] for b in ref['binary'] if b['head'] == 'softmax_MEL')),
            'softmax_MEL_AUC': float(next(b['roc_auc'] for b in ref['binary'] if b['head'] == 'softmax_MEL'))})
    assert len(identities) == 12
    index = {(r['model'], r['training_seed'], r['head'], r['metric']): r['value'] for r in metrics}
    assert len(index) == len(metrics)
    differences = []
    for r in metrics:
        if r['model'] == 'R_CLIN':
            for other in ['M0', 'R_IMG', 'XATTN']:
                key = (other, r['training_seed'], r['head'], r['metric'])
                if key in index:
                    differences.append({'contrast': f'R_CLIN-minus-{other}', 'training_seed': r['training_seed'],
                        'head': r['head'], 'metric': r['metric'], 'value': r['value'] - index[key]})
    summaries = summarize(metrics, ['model', 'head', 'metric'])
    paired_summary = summarize(differences, ['contrast', 'head', 'metric'])
    final_status = 'COMPLETE' if completed == 6 else ('BLOCKED' if errors else 'PARTIAL')
    for name, rows in [('RUN_IDENTITIES', identities), ('PER_SEED_VALIDATION_METRICS', metrics),
        ('THREE_SEED_VALIDATION_SUMMARY', summaries), ('PER_SEED_PAIRED_DIFFERENCES', differences),
        ('PAIRED_DIFFERENCE_SUMMARY', paired_summary), ('SELECTED_BEST_READOUT_DIAGNOSTICS', diagnostics)]:
        table(OUT / (name + '.csv'), rows)
    dump(OUT / 'P2_PROGRESS.json', {'status': final_status, 'updated_at': datetime.now().astimezone().isoformat(),
        'window': window, 'authorized_new_runs': 6, 'new_runs_training_and_validation_complete': completed,
        'P1_reused_runs': 3, 'M0_readonly_reference_runs': 3, 'new_training_count_is_not_12': True,
        'runs': identities, 'automatic_next_phase': False, 'next_step': 'STOP and return deliverables',
        'forbidden_evaluation_roles': ['HAM_test', 'SIIM', 'new_external']})
    for name in ['training', 'model', 'checks', 'finish']:
        before, after = Path(window['source_P1']) / f'p1_{name}.py', OUT / f'p2_{name}.py'
        diff = ''.join(difflib.unified_diff(before.read_text(encoding='utf-8').splitlines(True),
            after.read_text(encoding='utf-8').splitlines(True), fromfile=str(before), tofile=after.name))
        (OUT / f'P1_to_P2_{name}.diff').write_text(diff, encoding='utf-8')
    def num(r, k):
        return f'{r[k]:.6f}' if k in r else 'N/A'
    lines = '\n'.join('| ' + ' | '.join([r['run'], r['origin'], r['training_status'],
        str(r['completed_epochs']), str(r.get('selected_epoch')), num(r, 'macro_f1'), num(r, 'softmax_MEL_AP')]) + ' |' for r in identities)
    mean_lines = '\n'.join(f"| {r['model']} | {r['head']} | {r['metric']} | {r['n_seeds']} | {r['mean']:.6f} | {r['sample_SD_ddof1'] if r['sample_SD_ddof1'] is not None else 'N/A'} |"
        for r in summaries if (r['head'], r['metric']) in [('seven_class', 'macro_f1'), ('softmax_MEL', 'average_precision')])
    error_lines = '\n'.join(f"- {p.name}: {read(p).get('error', '').strip().splitlines()[-1]}" for p in errors) or '- No unhandled exception was recorded.'
    report = f'''# P2 Frozen-v1 Three-Seed Training Report

## Status: {final_status}
New P2 runs with completed training and selected-best validation: **{completed}/6**.
Separately reused: **3 P1 seed42 runs**, no retraining, inference or best change.
Read-only references: **3 current FIM_LPAH_FULL M0 runs**, HAM_val/NATIVE only.
These are not 12 new training runs. All prespecified candidates are retained irrespective of score.

| Run | Origin | Training status | Epochs | Best epoch | Val macro-F1 | Softmax MEL AP |
|---|---|---|---:|---:|---:|---:|
{lines}

## Per-run aggregation
| Model | Head | Metric | n seeds | Mean | Sample SD (ddof=1) |
|---|---|---|---:|---:|---:|
{mean_lines}

PER_SEED_VALIDATION_METRICS.csv contains both heads' AP/AUC/Brier/NLL and seven-class macro-F1/
Brier/NLL. THREE_SEED_VALIDATION_SUMMARY.csv gives arithmetic mean and sample SD of per-run
metrics, without pooled predictions/probability averaging. PER_SEED_PAIRED_DIFFERENCES.csv and
PAIRED_DIFFERENCE_SUMMARY.csv give R_CLIN-minus-M0/R_IMG/XATTN within each seed, then mean/SD.
Negative differences are favorable for lower-is-better losses. Partial groups show actual n/seeds.
No bootstrap,significance test,CI,threshold selection or calibration fitting was performed.

## Sources and frozen v1
- P1: `{window['source_P1']}`
- Original FIM seed42: `{window['source_seed42']}`
- Original FIM three-seed: `{window['source_three_seed']}`
- New output: `{OUT}`

Original region_readout_reference.py and all supplied original model/data/transform source are
byte-identical to P1. Region definitions,detach,residual path,CMGF,XATTN and losses are unchanged.
P2_PROTOCOL.json retains original mechanism/data/settings and specifies the new authorized scope.
P1 protocol is preserved separately; P1_to_P2_*.diff records local integration changes.

Seeds2024/3407 load their corresponding original pre-training initial_state_s{{seed}}.pt only.
P2_INITIALIZATION_s*.json and EXECUTION_FREEZE.json record original INITIALIZATION metadata,
size,SHA-256 and sources. No trained historical best/last or seed42 state initializes a new seed.
All three same-seed candidates strictly share base tensors. Isolated new-module seeds3024/4407
preserve CPU/CUDA RNG; R_IMG/R_CLIN initial readout hashes match. *_INITIALIZATION_CHECK.json
records base-key identity,actual imported source paths and new-layer hashes. Technical states are discarded.

Python/NumPy/Torch CPU/CUDA use training_seed. Classification order:seed+zero-based epoch;
per-image augmentation:seed+epoch*1000003+dataset_index*1009; auxiliary PCG64:
SeedSequence([seed,epoch,19]); DataLoader generator:seed*1000+17. Config/checkpoints carry the
training seed and fixed split42. Forbidden test/external loader assertions remain enforced.

## Actual protocol and checks
Fixed HAM6982 train/1513 validation,original22-dimensional clinical encoding,unchanged auxiliary100.
Prior P1 pairing/overlap/mechanism checks are reused,not a repeated project audit or old-ID search.
224 train; validation resize256/center224; original augmentations/normalization. Effective48/aux32,
micro8,workers2; AdamW backbone3e-5/other3e-4,wd0.05,warmup5,cosine80,max80,patience15,clip1.
Mask0.1; weighted CE beta0.999/smoothing0,BCE0.5,consistency0.05,sparse0.0005,TV0.002 unchanged.
FP16 AMP train,FP32 validation,TF32 off; overflow restores buffers/RNG and retries the same update.
Best selection:macro-F1,then softmax MEL AUC,then earliest exact tie. AP is logged,not used to
change checkpoint selection,budget or stopping. CUDA_TECHNICAL_CHECK_s*.json records actual
initialization,shape,gradient and full48+32 AMP checks with live two-worker loaders.
Regional validity/area/alpha,eta,gradient norm and training auxiliary Dice are saved per epoch.
Auxiliary Dice is an optimization diagnostic,not an independent localization test. XATTN has no regional alpha.

## Validation and M0
Selected-best HAM_val/NATIVE only:1513 images,164 MEL. Both heads:average_precision_score
(not trapezoidal PR area),ROC-AUC,unclipped Brier,binary NLL clipping[1e-7,1-1e-7]. Seven classes:
macro-F1,sum-over-seven Brier,NLL clipping[1e-7,1]. Float64 metric arrays match P1. Selected-best
F1/AUC/AP must reproduce training validation within1e-7. M0 is current FIM FULL,not ZERO or old B6;
only FULL/HAM_val/NATIVE aggregate rows are consumed. M0_VALIDATION_REFERENCES.json records
precise paths,selected rows and existing best identities. No M0 weights are loaded or inferred anew.

## Timing and incidents
Window:`{window['started_at']}` to `{window['deadline']}`,not P1's deadline.
D free at admission:{window['D_free_bytes']/1024**3:.2f}GiB; planning requirement8GiB.
RUN_IDENTITIES.csv gives completed-epoch time totals and run-wall time; epoch durations include
train/validation but exclude most checkpoint writes. TWO_EPOCH_PROGRESS.json gives early ETAs.
Unfinished runs retain their actual status. Last checkpoints contain optimizer,scheduler,scaler,
Python/NumPy/Torch CPU/CUDA RNG,data position,partial loss/diagnostics,history and best selection.
{error_lines}

## Interpretation and delivery
These are development-validation results across fixed training seeds,not independent external
evidence. Sample SD describes training-seed variation,not a clinical CI. All results are retained.
ZIP contains allowlisted protocol/code/diffs/configs/identities/logs/aggregate metrics/diagnostics.
No weights,images,per-case predictions,clinical rows or credentials. No original manuscript/result/
environment/hibernation/page-file change; no cloud,other direction,simplified-B2 training,test/SIIM
inference or automatic next stage. Return the requested three files and STOP.
'''
    (OUT / 'P2_TRAIN_REPORT.md').write_text(report, encoding='utf-8')
    names = ['P2_TRAIN_REPORT.md', 'P2_PROGRESS.json', 'WINDOW.json', 'P2_PROTOCOL.json',
        'EXECUTION_FREEZE.json', 'DATA_CHECK.json', 'M0_VALIDATION_REFERENCES.json',
        'P2_INITIALIZATION_s2024.json', 'P2_INITIALIZATION_s3407.json', 'training.log', 'RESOURCE_LOG.jsonl',
        'RUN_IDENTITIES.csv', 'PER_SEED_VALIDATION_METRICS.csv', 'THREE_SEED_VALIDATION_SUMMARY.csv',
        'PER_SEED_PAIRED_DIFFERENCES.csv', 'PAIRED_DIFFERENCE_SUMMARY.csv', 'SELECTED_BEST_READOUT_DIAGNOSTICS.csv']
    files = [OUT / n for n in names if (OUT / n).exists()]
    for pattern in ['p2_*.py', 'P1_to_P2_*.diff', '*_INITIALIZATION_CHECK.json', 'CUDA_TECHNICAL_CHECK_s*.json',
                    'PROCESS_*.json', 'console*.log', 'stderr*.log', 'PARAMETERIZATION_CHECK.json']:
        files += sorted(OUT.glob(pattern))
    files += errors
    for seed in [42, *SEEDS]:
        for variant in VARIANTS:
            root = (OUT / 'reused_P1' if seed == 42 else OUT) / f'{variant}_s{seed}'
            for name in ['effective_config.json', 'status.json', 'training_history.csv', 'readout_diagnostics.csv',
                         'TWO_EPOCH_PROGRESS.json', 'VALIDATION_RESULT.json']:
                if (root / name).exists():
                    files.append(root / name)
    files += [PKG / n for n in ['03_P1_PROTOCOL.json', 'region_readout_reference.py', 'test_region_readout_reference.py',
        '05_模型接入说明.md', 'Reference_Only/B6_seed42_original_config.yaml']]
    files += sorted((PKG / 'Reference_Only/model_code').rglob('*.py'))
    assert len(files) == len(set(files))
    assert all(p.suffix.lower() not in ['.pt', '.pth', '.npy', '.npz', '.jpg', '.png', '.env'] for p in files)
    assert not any('fim_mask_train100' in p.name for p in files)
    manifest = [{'relative_path': p.relative_to(OUT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
    dump(OUT / 'DELIVERY_CONTENTS.json', manifest)
    files.append(OUT / 'DELIVERY_CONTENTS.json')
    archive = OUT / 'P2_TRAIN_RESULTS.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(OUT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for item in manifest:
            assert hashlib.sha256(z.read(item['relative_path'])).hexdigest() == item['sha256']
    dump(OUT / 'ZIP_VERIFICATION.json', {'status': 'PASSED', 'path': str(archive), 'sha256': sha(archive),
        'members': len(files), 'bytes': archive.stat().st_size, 'CRC_and_each_SHA256': 'PASSED'})
    print(json.dumps({'status': final_status, 'new_completed': completed, 'reused_P1': 3, 'M0_readonly': 3,
        'report': str(OUT / 'P2_TRAIN_REPORT.md'), 'zip': str(archive), 'progress': str(OUT / 'P2_PROGRESS.json')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-only', action='store_true')
    args = parser.parse_args()
    if not args.package_only:
        for seed in SEEDS:
            for variant in VARIANTS:
                evaluate_run(seed, variant)
    package()
