"""Validation-only selected-best evaluation and allowlisted P1 delivery."""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import difflib
import hashlib
import json
import time
import zipfile

OUT = Path(__file__).resolve().parent
PKG = OUT / 'FIM_RegionClinical_Dev_Start'
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
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def evaluate():
    import gc
    import numpy as np
    import torch
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
    import p1_training as t
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert torch.cuda.is_available()
    val = t.HAMDataset(t.config(42), 'val')
    assert len(val) == 1513
    for variant in VARIANTS:
        root = OUT / f'{variant}_s42'
        status_path = root / 'status.json'
        if not status_path.exists() or read(status_path)['status'] != 'TRAINING_COMPLETE':
            continue
        if (root / 'VALIDATION_RESULT.json').exists():
            assert read(root / 'VALIDATION_RESULT.json')['best_sha256'] == sha(root / 'best.pt')
            continue
        if time.time() >= t.DEADLINE:
            t.log('PARTIAL: active deadline reached before remaining validation summary')
            return
        t.seed_all(42)
        state = torch.load(root / 'best.pt', map_location='cpu', weights_only=False)
        cfg = state['config']['original_config']
        model = t.fresh_model(cfg)
        model.load_state_dict(state['model'], strict=True)
        model.eval()
        expected = state['selection_validation']
        best_epoch = state['epoch']
        assert best_epoch == read(status_path)['best_epoch']
        del state
        probabilities, endpoints, targets = [], [], []
        diagnostics = {}
        started = time.perf_counter()
        with torch.inference_mode():
            for images, metadata, labels, _ in t.loader(val, list(range(len(val))), batch_size=8):
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
        result = {'status': 'COMPLETE', 'run': variant + '_s42', 'cohort': 'HAM_val', 'input': 'NATIVE',
            'best_path': str(root / 'best.pt'), 'best_sha256': sha(root / 'best.pt'), 'selected_epoch': best_epoch,
            'dtype': 'FP32', 'TF32': False, 'strict_loading': True, 'multiclass': metrics, 'binary': binary,
            'selection_metrics_reproduced': True, 'readout_diagnostics': t.summarize_diagnostics(diagnostics),
            'eta_raw': float(model.readout.eta.detach()), 'evaluation_seconds': time.perf_counter() - started,
            'per_example_predictions_exported': False}
        dump(root / 'VALIDATION_RESULT.json', result)
        t.log(f'VALIDATION_SUMMARY {variant} selected_epoch={best_epoch} AP={binary[0]["average_precision"]:.6f}')
        del model, probabilities, endpoints, targets, prob, endpoint
        gc.collect()
        torch.cuda.empty_cache()


def package():
    assert not (OUT / 'P1_DEV_RESULTS.zip').exists(), 'Do not overwrite a prior delivery'
    window = read(OUT / 'WINDOW.json')
    errors = sorted(OUT.glob('TRAINING_ERROR*.json')) + sorted(OUT.glob('FINALIZATION_ERROR*.json'))
    resumes = sorted(OUT.glob('PROCESS_RESUME_*.json'))
    baseline_path = OUT / 'M0_VALIDATION_REFERENCE.json'
    baseline = read(baseline_path) if baseline_path.exists() else None
    baseline_multi = baseline['INTERNAL_MULTICLASS_METRICS.csv']['selected_rows_only'][0] if baseline else {}
    baseline_binary = {r['head']: r for r in baseline['METRICS.csv']['selected_rows_only']} if baseline else {}
    summary, binary_rows, diag_rows = [], [], []
    for variant in VARIANTS:
        root = OUT / f'{variant}_s42'
        status = read(root / 'status.json') if (root / 'status.json').exists() else {'status': 'NOT_STARTED', 'completed_epochs': 0}
        row = {'run': variant + '_s42', 'status': status['status'], 'completed_epochs': status['completed_epochs'],
            'selected_epoch': status.get('best_epoch'), 'validation_summary_complete': False}
        if (root / 'effective_config.json').exists():
            cfg = read(root / 'effective_config.json')
            row.update(parameter_count=cfg['parameter_count'], new_readout_parameter_count=cfg['readout_parameter_count'])
        if (root / 'training_history.csv').exists():
            with (root / 'training_history.csv').open(encoding='utf-8', newline='') as f:
                history = list(csv.DictReader(f))
            row['sum_epoch_train_validation_seconds'] = sum(float(h['epoch_seconds']) for h in history)
            row['run_wall_seconds_in_last_invocation'] = status.get('elapsed_this_invocation_seconds')
        if (root / 'VALIDATION_RESULT.json').exists():
            result = read(root / 'VALIDATION_RESULT.json')
            assert result['best_sha256'] == sha(root / 'best.pt')
            row.update(validation_summary_complete=True, best_sha256=result['best_sha256'], **result['multiclass'])
            primary = result['binary'][0]
            row.update(val_softmax_MEL_AP=primary['average_precision'], val_softmax_MEL_AUC=primary['roc_auc'],
                delta_AP_vs_M0=primary['average_precision'] - float(baseline_binary['softmax_MEL']['average_precision']),
                delta_macro_f1_vs_M0=result['multiclass']['macro_f1'] - float(baseline_multi['macro_f1']))
            for b in result['binary']:
                b_row = {'run': variant + '_s42', **b}
                for key in ['roc_auc', 'average_precision', 'brier', 'binary_nll']:
                    b_row['delta_' + key + '_vs_M0'] = b[key] - float(baseline_binary[b['head']][key])
                binary_rows.append(b_row)
            diag_rows.append({'run': variant + '_s42', 'selected_epoch': result['selected_epoch'],
                'eta_raw': result['eta_raw'], 'regional_diagnostics_applicable': variant != 'XATTN', **result['readout_diagnostics']})
        summary.append(row)
    completed = sum(r['status'] == 'TRAINING_COMPLETE' and r['validation_summary_complete'] for r in summary)
    final_status = 'COMPLETE' if completed == 3 else ('BLOCKED' if errors else 'PARTIAL')
    table(OUT / 'VALIDATION_SUMMARY.csv', summary)
    table(OUT / 'VALIDATION_BINARY_METRICS.csv', binary_rows)
    table(OUT / 'SELECTED_BEST_READOUT_DIAGNOSTICS.csv', diag_rows)
    dump(OUT / 'P1_PROGRESS.json', {'status': final_status, 'updated_at': datetime.now().astimezone().isoformat(),
        'active_window': window, 'planned_new_runs': 3, 'completed_with_validation': completed, 'runs': summary,
        'M0': 'Existing current FIM FULL_s42 HAM_val/NATIVE only; not retrained or newly inferred',
        'next_step': 'STOP: return this package to user; no automatic next phase.',
        'automatic_next_phase': False, 'forbidden_eval_roles': ['HAM_test', 'SIIM', 'new_external']})
    receipt = read(OUT / 'EXTRACTION_RECEIPT.json')
    zip_path = Path(window['source_zip'])
    assert sha(zip_path) == receipt['zip_sha256']
    source_rows = []
    for rel in ['p1_training.py', 'p1_model.py', 'p1_checks.py', 'p1_finish.py', 'unpack_p1.py']:
        p = OUT / rel
        source_rows.append({'path': str(p), 'sha256': sha(p), 'type': 'new_local_code'})
    source_rows.append({'path': str(zip_path), 'sha256': sha(zip_path), 'type': 'unchanged_start_ZIP'})
    for rel in ['region_readout_reference.py', '03_P1_PROTOCOL.json', 'Reference_Only/model_code/lesionmamba/models/oncoderm_mambax.py']:
        p = PKG / rel
        expected = next(r['sha256'] for r in receipt['extracted_files'] if r['member'] == p.relative_to(OUT).as_posix())
        assert sha(p) == expected
        source_rows.append({'path': str(p), 'sha256': sha(p), 'type': 'unchanged_supplied_reference'})
    if (OUT / 'DATA_CHECK.json').exists():
        for rec in read(OUT / 'DATA_CHECK.json')['input_files']:
            assert sha(rec['path']) == rec['sha256']
            source_rows.append({**rec, 'type': 'unchanged_frozen_input'})
    table(OUT / 'SOURCE_IDENTITY.csv', source_rows)
    original_trainer = PKG / 'Reference_Only/fim_training_reference_DO_NOT_RUN.py'
    assert sha(original_trainer) == sha(Path(window['source_three_seed']) / 'fim_training.py')
    diff = ''.join(difflib.unified_diff(original_trainer.read_text(encoding='utf-8').splitlines(True),
        (OUT / 'p1_training.py').read_text(encoding='utf-8').splitlines(True),
        fromfile='immutable_final_three_seed_training_reference.py', tofile='p1_training.py'))
    (OUT / 'TRAINER_INTEGRATION.diff').write_text(diff, encoding='utf-8')
    def number(row, key):
        return f'{row[key]:.6f}' if key in row and row[key] is not None else 'N/A'
    lines = '\n'.join('| ' + ' | '.join([r['run'], r['status'], str(r['completed_epochs']), str(r.get('selected_epoch', 'N/A')),
        number(r, 'macro_f1'), number(r, 'val_softmax_MEL_AUC'), number(r, 'val_softmax_MEL_AP'), number(r, 'delta_AP_vs_M0')]) + ' |' for r in summary)
    m0_text = (f"Existing M0 FULL_s42: selected epoch {baseline['best_identity']['best_epoch']}, "
        f"macro-F1={baseline_multi['macro_f1']}, MEL AUC={baseline_binary['softmax_MEL']['roc_auc']}, "
        f"MEL AP={baseline_binary['softmax_MEL']['average_precision']}. "
        "Only original HAM_val/NATIVE aggregate records were extracted; no M0 retraining or new inference."
        if baseline else 'M0 reference extraction did not finish; no reference numbers are invented.')
    incident_text = '\n'.join(f'- `{p.name}`: `{read(p).get("error", "").strip().splitlines()[-1]}`' for p in errors) or '- No exception was recorded in this invocation.'
    for p in resumes:
        resume = read(p)
        incident_text += (f"\n- `{p.name}`: earlier execution process ended without a Python traceback. "
            f"Resumed from complete epoch{resume['last_verified_completed_epoch']} optimizer/scheduler/scaler/RNG checkpoint "
            f"at {resume['started_at']}; completed epochs were not rerun. The interrupted uncheckpointed epoch was "
            "recovered from the preceding complete checkpoint. Cause of the process exit is unconfirmed; "
            "this was not early stopping and the active deadline was not extended.")
    report = f"""# P1 Development Report

## Actual status: {final_status}

Completed prescribed training and selected-best validation: {completed}/3.
All three prespecified candidates are retained below, irrespective of result direction.
No uncompleted run is described as finished. No P2 or external evaluation is authorized or started.

| Candidate | Training status | Epochs completed | Selected epoch | Val macro-F1 | Val MEL AUC | Val MEL AP | AP minus M0 |
|---|---|---:|---:|---:|---:|---:|---:|
{lines}

{m0_text}

`VALIDATION_BINARY_METRICS.csv` reports both heads' AP/AUC, binary Brier and binary NLL, and
candidate-minus-M0 differences. `VALIDATION_SUMMARY.csv` additionally reports seven-class macro-F1,
sum-over-seven-classes Brier, multiclass NLL, parameter counts and training times where available.
NLL clips probabilities to [1e-7,1-1e-7] for binary and [1e-7,1] for multiclass, matching the existing
reference statistic. AP is sklearn average_precision_score, not trapezoidal PR area.
Metrics are computed per image on the same 1,513 HAM validation images (164 MEL), without bootstrap,
calibration fitting, threshold selection, confidence intervals or significance claims.

## Sources and isolation

- Original ZIP: `{window['source_zip']}`. CRC passed; original ZIP retained unchanged.
- This output: `{OUT}`. D free at admission: {window['D_free_bytes']/1024**3:.2f} GiB;
  working estimate6 GiB for three best/last checkpoints plus temporary save space.
- Seed42 original input: `{window['source_seed42']}`.
- Final three-seed training reference: `{window['source_three_seed']}`.
- Shared initial state: original `initial_state_s42.pt`, SHA-256
  `cb1dc95021bb234e039a1819ab1aca3519c175fc0dc9555b1963c5914d0b8fcc`.
- Only start-ZIP/new-output instructions were redirected to D in the extracted guides.
  C-drive read-only input paths and the scientific protocol were not globally replaced.
- Actual source imports and strict base key maps are recorded in `*_INITIALIZATION_CHECK.json`;
  source hashes and unchanged frozen inputs are listed in `SOURCE_IDENTITY.csv`.
- The candidate reference module and original model module remain byte-identical to the supplied ZIP.
  `TRAINER_INTEGRATION.diff` records the isolated trainer adaptation.

## Preserved training protocol

Training seed42, HAM split42, 6,982 train/1,513 validation and the exact frozen auxiliary100 membership.
Original 22-dimensional clinical encoding, classification augmentation,224 input; validation resize256,
center224. All candidate base tensors strictly match the original pre-training state before new module
initialization. New parameters use isolated seed1042; R_IMG/R_CLIN readout initial states match exactly.
No trained original best/last is used to initialize any candidate.

Classification effective48, auxiliary32, micro8, workers2; original AdamW backbone3e-5/other3e-4,
wd0.05, warmup5, cosine80, max80, patience15 and clip1. FP16 AMP train, FP32 validation, TF32 off.
Class-balanced CE beta0.999 and effective smoothing0; endpoint BCE0.5, consistency0.05,
attention sparse0.0005, TV0.002, auxiliary Dice+BCE0.1 for every candidate. Existing AMP overflow
fallback restores RNG/BN and retries the same effective update instead of skipping it.
Best is selected by validation macro-F1, then seven-class MEL AUC, then earlier exact tie.
AP is logged each epoch but does not replace this checkpoint-selection rule.

## Checks and diagnostics

Supplied nine CPU tests and the recorded CUDA integration gate cover eta0 identity, strict initialization,
one backbone/fuse/mask/metadata call, nonzero-eta gradients, detached predicted mask weights,
live auxiliary-mask gradients, original CMGF, R_IMG metadata-invariant alpha and R_CLIN-responsive alpha,
constant-mask finiteness, weighted-CE microbatch reduction and forbidden-loader assertions.
The new CUDA gate keeps both DataLoader workers alive during complete effective48+aux32 updates.
All technical-test states are discarded before formal training.

Per-epoch `readout_diagnostics.csv` reports region validity/area, alpha mean/within-sample SD,
residual eta/coefficient, gradient norm and auxiliary soft Dice where training occurred.
`SELECTED_BEST_READOUT_DIAGNOSTICS.csv` summarizes selected-best validation readouts where complete.
XATTN has no three-region alpha; these regional measures are not applicable, not zero.
Training auxiliary Dice is an optimization diagnostic, not independent localization performance.
The within-sample SD is descriptive dispersion, not uncertainty across independent training seeds.

## Timing, resource observations and incidents

Fresh active window: `{window['started_at']}` through `{window['deadline']}`.
Original prior-blocked directory remains untouched: `{window.get('previous_blocked_directory', '')}`.
`TWO_EPOCH_PROGRESS.json` records each available early ETA. Epoch durations include training and
validation but exclude most checkpoint write time; the separate run-wall field includes run management.
The specified80-epoch maximum is not shortened to force completion. Early stopping uses patience15.

{incident_text}

`RESOURCE_LOG.jsonl` records Windows physical-memory/commit headroom and CUDA allocation during the
live-worker gate. Hibernation/page-file/global settings were not modified and no other process was killed.
No remote/cloud resources were used. Unfinished runs retain their actual status/checkpoint, never a
fabricated best or performance result.

## Interpretation and delivery

This is seed42 development on a repeatedly consulted validation set, not independent held-out evidence.
Differences from the reused M0 reference are descriptive. A favorable validation AP difference is not an
external improvement, a significance finding or an acceptance-probability claim. No candidate or epoch
is removed because of a low score. Decisions about additional seeds/P2 remain with the user.

The ZIP is built from an explicit allowlist of code/configuration, hashes, logs, aggregate metrics,
diagnostics, scoped checks and this report. It excludes weights/optimizer checkpoints, images/masks,
per-example predictions, private membership/clinical rows, credentials and environment secrets.
No HAM test, SIIM, new external cohort, B1/B2 rerun, bootstrap, other research line or manuscript edits.
Return `P1_DEV_REPORT.md`, `P1_DEV_RESULTS.zip`, `P1_PROGRESS.json`, then stop.
"""
    (OUT / 'P1_DEV_REPORT.md').write_text(report, encoding='utf-8')
    names = ['P1_DEV_REPORT.md', 'P1_PROGRESS.json', 'WINDOW.json', 'EXTRACTION_RECEIPT.json',
        'DATA_CHECK.json', 'P1_INITIALIZATION.json', 'M0_VALIDATION_REFERENCE.json', 'CPU_TEST_RESULT.json',
        'CUDA_TECHNICAL_CHECK.json', 'RESOURCE_LOG.jsonl', 'SOURCE_IDENTITY.csv', 'TRAINER_INTEGRATION.diff',
        'p1_training.py', 'p1_model.py', 'p1_checks.py', 'p1_finish.py', 'unpack_p1.py', 'training.log',
        'VALIDATION_SUMMARY.csv', 'VALIDATION_BINARY_METRICS.csv', 'SELECTED_BEST_READOUT_DIAGNOSTICS.csv']
    files = [OUT / n for n in names if (OUT / n).exists()]
    files += sorted(OUT.glob('*_INITIALIZATION_CHECK.json')) + errors + resumes
    files += sorted(OUT.glob('console_resume_*.log')) + sorted(OUT.glob('stderr_resume_*.log'))
    for variant in VARIANTS:
        root = OUT / f'{variant}_s42'
        for name in ['effective_config.json', 'status.json', 'training_history.csv', 'readout_diagnostics.csv',
                     'TWO_EPOCH_PROGRESS.json', 'VALIDATION_RESULT.json']:
            if (root / name).exists():
                files.append(root / name)
    files += [PKG / n for n in ['03_P1_PROTOCOL.json', 'region_readout_reference.py',
        'test_region_readout_reference.py', '05_模型接入说明.md', 'Reference_Only/B6_seed42_original_config.yaml']]
    files += sorted((PKG / 'Reference_Only/model_code').rglob('*.py'))
    assert len(files) == len(set(files))
    assert all(p.suffix.lower() not in ['.pt', '.pth', '.npy', '.npz', '.jpg', '.png', '.env'] for p in files)
    assert not any('fim_mask_train100' in p.name for p in files)
    manifest = [{'relative_path': p.relative_to(OUT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
    dump(OUT / 'DELIVERY_CONTENTS.json', manifest)
    files.append(OUT / 'DELIVERY_CONTENTS.json')
    archive = OUT / 'P1_DEV_RESULTS.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(OUT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for item in manifest:
            assert hashlib.sha256(z.read(item['relative_path'])).hexdigest() == item['sha256']
    dump(OUT / 'ZIP_VERIFICATION.json', {'status': 'PASSED', 'path': str(archive), 'sha256': sha(archive),
        'members': len(files), 'bytes': archive.stat().st_size, 'CRC_and_each_SHA256': 'PASSED'})
    print(json.dumps({'status': final_status, 'completed_runs': completed,
        'report': str(OUT / 'P1_DEV_REPORT.md'), 'zip': str(archive), 'progress': str(OUT / 'P1_PROGRESS.json')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-only', action='store_true')
    args = parser.parse_args()
    if not args.package_only:
        try:
            evaluate()
        except Exception:
            import traceback
            error = traceback.format_exc()
            dump(OUT / f'FINALIZATION_ERROR_{datetime.now():%Y%m%d_%H%M%S}.json', {'error': error})
            print(error, flush=True)
    package()
