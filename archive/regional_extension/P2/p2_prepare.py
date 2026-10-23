"""Scoped, read-only P1 reuse and two specified pre-training state identities."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
from datetime import datetime

OUT = Path(__file__).resolve().parent
WINDOW = json.loads((OUT / 'WINDOW.json').read_text(encoding='utf-8-sig'))
P1 = Path(WINDOW['source_P1'])
THREE = Path(WINDOW['source_three_seed'])
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


def record(path, role):
    path = Path(path)
    return {'path': str(path), 'role': role, 'sha256': sha(path), 'bytes': path.stat().st_size}


def full_validation_rows(path):
    # Do not parse or consume other conditions/cohorts' numeric results.
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        header = next(csv.reader([next(f)]))
        selected = []
        for line in f:
            if ',FULL,HAM_val,' not in line:
                continue
            row = dict(zip(header, next(csv.reader([line]))))
            if row['condition'] == 'FULL' and row['cohort'] == 'HAM_val' and row.get('input', 'NATIVE') == 'NATIVE':
                selected.append(row)
        return selected


def verify_freeze():
    freeze = read(OUT / 'EXECUTION_FREEZE.json')
    for rec in freeze['files']:
        assert sha(rec['path']) == rec['sha256'], f"Frozen input/code changed: {rec['path']}"


def prepare():
    if (OUT / 'EXECUTION_FREEZE.json').exists():
        verify_freeze()
        print('Existing P2 freeze verified; not replaced.')
        return
    assert shutil.disk_usage(OUT).free >= 8 * 1024**3
    records = []
    for name in ['P1_DEV_REPORT.md', 'P1_PROGRESS.json', 'p1_training.py', 'p1_model.py', 'p1_checks.py',
                 'p1_finish.py', 'DATA_CHECK.json', 'CUDA_TECHNICAL_CHECK.json', 'CPU_TEST_RESULT.json']:
        records.append(record(P1 / name, 'read_only_P1'))
    assert read(P1 / 'P1_PROGRESS.json')['status'] == 'COMPLETE'
    assert read(P1 / 'CUDA_TECHNICAL_CHECK.json')['status'] == 'PASSED'
    p1pkg = P1 / PKG.name
    for path in sorted(PKG.rglob('*')):
        if path.is_file():
            source = p1pkg / path.relative_to(PKG)
            assert sha(path) == sha(source), str(path)
            records.extend([record(source, 'unchanged_P1_reference'), record(path, 'byte_identical_P2_reference')])
    data = read(P1 / 'DATA_CHECK.json')
    assert data['status'] == 'PASSED' and data['aux'] == 100
    for rec in data['input_files']:
        if Path(rec['path']).name == 'initial_state_s42.pt':
            continue
        assert sha(rec['path']) == rec['sha256']
        records.append(record(rec['path'], 'unchanged_P1_data_identity'))
    assert sha(OUT / 'fim_mask_train100.csv') == data['selected_manifest_sha256']
    records.append(record(OUT / 'fim_mask_train100.csv', 'private_frozen_membership_not_for_ZIP'))
    dump(OUT / 'DATA_CHECK.json', {'status': 'REUSED_P1_PASSED', 'source': str(P1 / 'DATA_CHECK.json'),
        'source_sha256': sha(P1 / 'DATA_CHECK.json'), 'HAM_train': 6982, 'HAM_val': 1513, 'aux': 100,
        'selected_manifest_sha256': data['selected_manifest_sha256'], 'member_reselection': False,
        'prior_pairing_and_overlap_checks_reused': True, 'new_full_data_audit': False})
    initial_records = []
    for seed in SEEDS:
        jp, state = THREE / f'INITIALIZATION_s{seed}.json', THREE / f'initial_state_s{seed}.pt'
        init = read(jp)
        assert init['status'] == 'READY' and init['training_seed'] == seed and init['split_seed'] == 42
        assert init['initial_state_sha256'] == sha(state)
        assert init['snapshot_revision'] == 'aa096f03029c7f0ec052013f64c819b34f8ad790'
        assert init['public_weight_sha256'] == 'a1aefa409b513cf209b085424eb3efffe4e4a9f511491bc2c12ea35209e6bb95'
        records.extend([record(jp, 'original_seed_initialization'), record(state, 'original_pretraining_state_not_for_ZIP')])
        dump(OUT / f'P2_INITIALIZATION_s{seed}.json', {'base': init, 'training_seed': seed,
            'new_module_seed': seed + 1000, 'source_loaded': str(state), 'strict_base_load_required': True,
            'trained_weights_used_for_initialization': False, 'new_module_rng_isolated_required': True})
        initial_records.append(init)
    assert initial_records[0]['backbone_tensor_sha256'] == initial_records[1]['backbone_tensor_sha256']
    assert initial_records[0]['nonbackbone_tensor_sha256'] != initial_records[1]['nonbackbone_tensor_sha256']
    reused = OUT / 'reused_P1'
    reused.mkdir(exist_ok=False)
    for variant in VARIANTS:
        dst = reused / f'{variant}_s42'
        dst.mkdir()
        src = P1 / dst.name
        result, status = read(src / 'VALIDATION_RESULT.json'), read(src / 'status.json')
        assert result['status'] == 'COMPLETE' and result['cohort'] == 'HAM_val' and result['input'] == 'NATIVE'
        assert status['status'] == 'TRAINING_COMPLETE' and result['selected_epoch'] == status['best_epoch']
        assert sha(src / 'best.pt') == result['best_sha256']
        records.append(record(src / 'best.pt', 'P1_best_hashed_only_not_loaded'))
        for name in ['VALIDATION_RESULT.json', 'effective_config.json', 'status.json', 'training_history.csv', 'readout_diagnostics.csv']:
            shutil.copy2(src / name, dst / name)
            records.extend([record(src / name, 'P1_seed42_reused_no_execution'), record(dst / name, 'P1_reused_copy')])
    m0 = read(P1 / 'M0_VALIDATION_REFERENCE.json')
    m0refs = [{'training_seed': 42, 'condition': 'FULL', 'cohort': 'HAM_val', 'input': 'NATIVE',
        'multiclass': m0['INTERNAL_MULTICLASS_METRICS.csv']['selected_rows_only'][0],
        'binary': m0['METRICS.csv']['selected_rows_only'], 'identity': m0['best_identity'],
        'status': m0['status'], 'source': str(P1 / 'M0_VALIDATION_REFERENCE.json')}]
    records.append(record(P1 / 'M0_VALIDATION_REFERENCE.json', 'M0_seed42_validation_only'))
    binary_path = THREE / 'ALL_METRICS.csv'
    multi_path = THREE / 'ALL_INTERNAL_MULTICLASS_METRICS.csv'
    binary_rows, multi_rows = full_validation_rows(binary_path), full_validation_rows(multi_path)
    for seed in SEEDS:
        binary = [r for r in binary_rows if int(r['training_seed']) == seed]
        multi = [r for r in multi_rows if int(r['training_seed']) == seed]
        assert len(binary) == 2 and len(multi) == 1
        assert {r['head'] for r in binary} == {'softmax_MEL', 'endpoint_MEL'}
        assert all(int(r['images']) == 1513 and int(r['positive_images']) == 164 for r in binary)
        identity_path = THREE / f'FULL_s{seed}_EVALUATION_MODEL.json'
        status_path = THREE / f'FIM_LPAH_FULL_s{seed}/status.json'
        identity, status = read(identity_path), read(status_path)
        assert status['status'] == 'TRAINING_COMPLETE' and identity['best_epoch'] == status['best_epoch']
        assert f'FIM_LPAH_FULL_s{seed}' in identity['path']
        m0refs.append({'training_seed': seed, 'condition': 'FULL', 'cohort': 'HAM_val', 'input': 'NATIVE',
            'binary': binary, 'multiclass': multi[0], 'identity': identity, 'status': status,
            'source_binary': str(binary_path), 'source_multiclass': str(multi_path),
            'source_identity': str(identity_path)})
        records.extend([record(identity_path, 'M0_recorded_identity_no_weight_load'), record(status_path, 'M0_completed_status')])
    dump(OUT / 'M0_VALIDATION_REFERENCES.json', {'references': m0refs,
        'selection': 'FULL / HAM_val / NATIVE only; other result rows ignored before numeric parsing',
        'new_M0_inference': False, 'ZERO_or_historical_B6_used': False})
    protocol = read(PKG / '03_P1_PROTOCOL.json')
    protocol['protocol'] = 'FIM_REGION_CLINICAL_P2_frozen_v1'
    protocol['status'] = 'FROZEN_BEFORE_TRAINING'
    protocol['scope'].update(phase='P2', new_training_runs=[f'{v}_s{s}' for s in SEEDS for v in VARIANTS],
        reference='current_FIM_FULL_s42_s2024_s3407_HAM_val_NATIVE_only', reused_P1_runs=[f'{v}_s42' for v in VARIANTS])
    protocol['training'].pop('seed')
    protocol['training'].update(training_seeds=SEEDS, new_module_seed_rule='training_seed + 1000')
    protocol['decision'] = {'checkpoint_rule_unchanged': True, 'mechanism_v1_frozen': True,
        'no_automatic_next_phase': True, 'summary': 'per-run metrics, arithmetic mean, sample SD(ddof=1), within-seed differences; no bootstrap'}
    protocol['delivery'] = ['P2_TRAIN_REPORT.md', 'P2_TRAIN_RESULTS.zip', 'P2_PROGRESS.json']
    dump(OUT / 'P2_PROTOCOL.json', protocol)
    for p in sorted(OUT.glob('p2_*.py')):
        records.append(record(p, 'new_P2_code'))
    for name in ['WINDOW.json', 'P2_PROTOCOL.json', 'M0_VALIDATION_REFERENCES.json', 'DATA_CHECK.json',
                 'P2_INITIALIZATION_s2024.json', 'P2_INITIALIZATION_s3407.json']:
        records.append(record(OUT / name, 'frozen_P2_record'))
    dump(OUT / 'EXECUTION_FREEZE.json', {'created_at': datetime.now().astimezone().isoformat(),
        'files': records, 'new_runs': 6, 'reused_P1_runs': 3, 'readonly_M0_runs': 3,
        'data_and_source_scan_scope': 'explicit required paths only'})
    print('P2 scoped freeze: PASS; six new runs, three P1 reuses, three M0 validation-only references.')


if __name__ == '__main__':
    prepare()
