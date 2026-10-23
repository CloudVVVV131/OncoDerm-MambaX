"""Scoped P6 source/data identities and read-only validation references."""
import ast
import csv
import difflib
from datetime import datetime
import shutil
from p6_common import OUT, PKG, OLD, P1, P2, P3, WINDOW, read, dump, sha, table


def prepare():
    assert shutil.disk_usage(OUT).free >= 4 * 1024**3
    input_records = read(P1 / 'DATA_CHECK.json')
    assert input_records['HAM_train'] == 6982 and input_records['HAM_val'] == 1513
    for item in input_records['input_files']:
        assert sha(item['path']) == item['sha256'], item['path']
    assert sha(OUT / 'fim_mask_train100.csv') == input_records['selected_manifest_sha256']
    assert sha(OLD / 'initial_state_s42.pt') == read(OLD / 'INITIALIZATION.json')['initial_state_sha256']
    copied = []
    for destination in (PKG / 'Reference_Only/model_code').rglob('*.py'):
        source = P1 / destination.relative_to(OUT)
        assert sha(source) == sha(destination)
        copied.append({'source': str(source), 'copy': str(destination), 'sha256': sha(destination)})
    for relative in ['region_readout_reference.py', 'Reference_Only/B6_seed42_original_config.yaml']:
        source, dest = P1 / 'FIM_RegionClinical_Dev_Start' / relative, PKG / relative
        assert sha(source) == sha(dest)
        copied.append({'source': str(source), 'copy': str(dest), 'sha256': sha(dest)})
    dump(OUT / 'SOURCE_COPY_IDENTITIES.json', copied)
    dump(OUT / 'DATA_CHECK.json', {**input_records, 'rechecked_at': datetime.now().astimezone().isoformat(),
        'previous_check_reused': str(P1 / 'DATA_CHECK.json'),
        'original_members_not_reselected': True, 'current_input_hashes_match': True})
    references = []
    for filename in ['PER_RUN_METRICS.csv', 'INTERNAL_MULTICLASS_METRICS.csv']:
        source = P3 / filename
        with source.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                # Only allowed aggregate rows are retained or used for comparisons.
                if row['cohort'] != 'HAM_val' or row['input'] != 'NATIVE':
                    continue
                if row['training_seed'] != '42' or row['model'] not in ['M0', 'R_IMG', 'R_CLIN', 'XATTN']:
                    continue
                references.append({**row, 'source_file': str(source), 'source_sha256': sha(source),
                                   'origin': 'READ_ONLY_EXISTING_SELECTED_BEST_VALIDATION'})
    assert len(references) == 12
    identities = read(P3 / 'FIXED_MODEL_IDENTITIES.json')
    if isinstance(identities, dict):
        identities = identities['models']
    chosen = [r for r in identities if r['training_seed'] == 42 and r['model'] in ['M0', 'R_IMG', 'R_CLIN', 'XATTN']]
    assert len(chosen) == 4
    dump(OUT / 'VALIDATION_REFERENCE_IDENTITIES.json', chosen)
    table(OUT / 'READ_ONLY_VALIDATION_REFERENCES.csv', references)
    protocol = {'version': 'P6-v2.0', 'task_sha256': sha(OUT / 'TASK_PROMPT.txt'),
        'window': WINDOW, 'frozen_at': datetime.now().astimezone().isoformat(),
        'conditions': {'FIM_TKEEP_REGION_s42': 'region', 'FIM_TKEEP_GLOBAL_s42': 'global'},
        'readout': {'channels': 768, 'hidden': 128, 'slot_order': ['g', 'interior', 'boundary', 'context'],
                    'slots': '4 independent LayerNorm(768),Linear(768,128,bias=True),GELU',
                    'fusion': 'concat -> LayerNorm(512) -> Linear(512,768,bias=False)',
                    'output': 'g + 0.5*tanh(eta)*(h-g)', 'eta_initial': 0,
                    'new_module_seed': 6042, 'clinical_in_readout': False,
                    'mask_weights_detached': True, 'min_mass_fraction': 1e-4,
                    'global_control': 'all three region slots use same fused feature GAP'},
        'training_seed': 42, 'split_seed': 42, 'new_training_runs': 2,
        'train_n': 6982, 'validation_n': 1513, 'aux_n': 100,
        'classification_order': '42+epoch', 'image_augmentation': '42+epoch*1000003+index*1009',
        'aux_rng': 'PCG64 SeedSequence([42,epoch,19])', 'loader_seed': 42017,
        'effective_batches': [48, 32], 'microbatch': 8, 'workers': 2,
        'optimizer': 'AdamW backbone3e-5/other3e-4,wd0.05', 'schedule': 'warmup5,cosine80',
        'max_epochs': 80, 'patience': 15, 'gradient_clip': 1,
        'loss': {'CE_beta': .999, 'smoothing_effective': 0, 'CE_denominator': 'all effective-batch target class weights',
                 'melanoma_BCE': .5, 'consistency': .05, 'mask_Dice_BCE': .1, 'sparsity': .0005, 'TV': .002},
        'best_selection': ['validation_macro_F1', 'validation_softmax_MEL_AUC', 'earlier_exact_tie'],
        'AP_used_for_selection': False, 'FP32_eval': True, 'TF32': False,
        'bootstrap': False, 'threshold_selection': False, 'calibrator_fitting': False,
        'external_evaluation': False, 'new_seed_expansion': False,
        'main_development_contrast': 'TKEEP_REGION-TKEEP_GLOBAL validation softmax_MEL AP',
        'status_of_hypothesis': 'post-P5 single-seed development, not independent confirmation'}
    dump(OUT / 'P6_FROZEN_PROTOCOL.json', protocol)
    for original, local in [('p2_training.py', 'p6_training.py'), ('p2_worker_data.py', 'p6_worker_data.py')]:
        diff = ''.join(difflib.unified_diff((P2 / original).read_text(encoding='utf-8').splitlines(True),
                    (OUT / local).read_text(encoding='utf-8').splitlines(True), fromfile=str(P2 / original), tofile=local))
        (OUT / (local + '.diff')).write_text(diff, encoding='utf-8')
    a = ast.parse((P2 / 'p2_training.py').read_text(encoding='utf-8'))
    b = ast.parse((OUT / 'p6_training.py').read_text(encoding='utf-8'))
    identical = ['loader', 'load_aux', 'optimizer_for', 'class_weights', 'dice_bce', 'update',
                 'seed_all', 'rng_state', 'restore_rng', 'validate']
    for name in identical:
        na = next(n for n in a.body if getattr(n, 'name', None) == name)
        nb = next(n for n in b.body if getattr(n, 'name', None) == name)
        assert ast.dump(na, include_attributes=False) == ast.dump(nb, include_attributes=False), name
    dump(OUT / 'TRAINER_REUSE_CHECK.json', {'identical_function_AST': identical,
        'loss_update_change': 'module gradient diagnostics only; original arithmetic preserved',
        'source': str(P2 / 'p2_training.py'), 'worker_entry': 'stdlib-only main; isolated dataset module'})


def freeze():
    files = list(OUT.glob('*.py')) + list(PKG.rglob('*.py')) + [OUT / 'P6_FROZEN_PROTOCOL.json', OUT / 'WINDOW.json',
            OUT / 'TASK_PROMPT.txt', PKG / 'Reference_Only/B6_seed42_original_config.yaml']
    dump(OUT / 'P6_CODE_FREEZE.json', [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files])


if __name__ == '__main__':
    prepare()
