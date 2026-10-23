"""Freeze every fixed best and the analysis before any P3 held-out forward pass."""
import ast
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from p3_common import *


def freeze():
    p2 = Path(WINDOW['source_P2'])
    p1 = Path(WINDOW['source_P1'])
    delivery = Path(WINDOW['source_P2_delivery'])
    prior = {r['run']: r for r in csv.DictReader((delivery / 'RUN_IDENTITIES.csv').open(encoding='utf-8-sig'))}
    refs = read(p2 / 'M0_VALIDATION_REFERENCES.json')['references']
    models = []
    for seed in SEEDS:
        for family in MODELS:
            if family == 'M0':
                r = next(r for r in refs if r['training_seed'] == seed)
                best = Path(r['identity']['path'])
            elif family == 'B2S':
                best = OUT / f'FIM_B2S_s{seed}/best.pt'
            else:
                best = (p1 if seed == 42 else p2) / f'{family}_s{seed}/best.pt'
            status = read(best.parent / 'status.json')
            assert status['status'] == 'TRAINING_COMPLETE', str(best)
            expected = prior.get(best.parent.name)
            digest = sha(best)
            if family != 'B2S':
                assert digest == expected['best_sha256']
                assert status['best_epoch'] == int(expected['selected_epoch'])
            cfg = best.parent / 'effective_config.json'
            snapshot = OUT / 'provenance' / 'fixed_run_records' / best.parent.name
            snapshot.mkdir(parents=True, exist_ok=True)
            for filename in ['effective_config.json', 'training_history.csv', 'status.json']:
                original = best.parent / filename
                destination = snapshot / filename
                if destination.exists():
                    assert sha(destination) == sha(original)
                else:
                    shutil.copy2(original, destination)
            models.append({'model': family, 'training_seed': seed, 'run': best.parent.name,
                'path': str(best), 'sha256': digest, 'bytes': best.stat().st_size,
                'best_epoch': status['best_epoch'], 'completed_epochs': status['completed_epochs'],
                'effective_config_path': str(cfg), 'effective_config_sha256': sha(cfg),
                'config_snapshot': str(snapshot / 'effective_config.json'),
                'origin': 'new_P3_B2S_training' if family == 'B2S' else 'existing_fixed_best_no_retraining',
                'prior_validation_reference': None if family == 'B2S' else {
                    k: float(expected[k]) for k in ['macro_f1','softmax_MEL_AP','softmax_MEL_AUC']}})
    # The candidate forward remains exactly the executed P1/P2 forward.
    new_tree = ast.parse((OUT / 'p3_eval_model.py').read_text())
    new = next(n for n in new_tree.body if getattr(n,'name','') == 'FixedCandidate')
    new_forward = next(n for n in new.body if getattr(n,'name','') == 'forward')
    for base, name, cls in [(p1,'p1_model.py','P1Model'),(p2,'p2_model.py','P2Model')]:
        tree = ast.parse((base / name).read_text())
        cl = next(n for n in tree.body if getattr(n,'name','') == cls)
        fn = next(n for n in cl.body if getattr(n,'name','') == 'forward')
        assert ast.dump(fn) == ast.dump(new_forward)
    assert sha(SOURCE / 'region_readout_reference.py') == sha(p2 / 'FIM_RegionClinical_Dev_Start/region_readout_reference.py')
    inputs = [ARCHIVE / f'splits/ham10000_seed42_{split}.csv' for split in ['train','val','test']]
    inputs += [ROOT / '_figure_source_freeze_20260819/ISIC_2020_Training_GroundTruth_v2.csv', SOURCE / 'metadata_mapping.json']
    code = sorted(OUT.glob('*.py')) + sorted(SOURCE.rglob('*.py'))
    manifest = [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in inputs + code]
    protocol = {
        'authorization': 'P3 only; old P1/P2 test-disabled records unchanged',
        'models': models, 'new_training_runs': 3, 'existing_forward_only_runs': 12,
        'candidate_mechanism': 'P1/P2 v1 unchanged; forward AST identical; reference readout bytes identical',
        'cohorts': {'HAM_val':1513, 'HAM_test':1520, 'SIIM':33126},
        'external_cohort_note': 'Previously used SIIM cohort, not a new independent external cohort',
        'SIIM_conditions':['E_REAL','E_NULL'], 'heads':['softmax_MEL','endpoint_MEL'],
        'precision':'FP32, eval, inference_mode, TF32 off', 'inference_batch':8,
        'preprocessing':'original RGB resize short256 / center224 / ImageNet normalization',
        'cache':'batch-local backbone/fused/mask only; recompute clinical encoder, readout and CMGF per condition; verify full forward atol1e-5 rtol1e-4 per best; otherwise full forward',
        'validation_reproduction_tolerance':1e-7,
        'primary_comparisons':['R_CLIN_minus_M0','R_CLIN_minus_XATTN'],
        'primary_endpoint':'mean of three per-seed SIIM E_REAL softmax_MEL average_precision differences',
        'supporting_comparisons':['R_CLIN_minus_R_IMG','R_CLIN_minus_B2S'],
        'interval_scope':'four prespecified comparisons x two input conditions x both heads x AP/AUC; primary only E_REAL/softmax AP for the first two comparisons',
        'metrics':{'AP':'sklearn average_precision_score, not trapezoidal PR area',
          'AUC':'sklearn roc_auc_score','Brier':'unclipped mean squared error',
          'binary_NLL':'epsilon1e-7, clipping [epsilon,1-epsilon]',
          'multiclass_Brier':'mean of sum over seven classes',
          'multiclass_NLL':'target probability clipped [1e-7,1]',
          'macro_F1':'labels 0..6, arithmetic macro',
          'binary_ECE15':'positive-probability equal-width bins [k/15,(k+1)/15), last includes1; sum n_bin/N*abs(mean_probability-observed_positive_rate)',
          'multiclass_ECE15':'top-label confidence/accuracy, same15bins; empty bins mean/rate null, contribution zero'},
        'aggregation':'per-model per-seed metrics first, then mean and sample SD ddof1; paired same-seed differences; no pooled predictions or probability ensemble',
        'bootstrap':{'draws':10000,'unit':'actual SIIM patient_id','patient_sort':'ascending lexical string via np.unique',
            'rng':'PCG64','seed':20261016,'paired_across':'all15models,all3seeds,bothconditions,bothheads',
            'resampling':'sample2056patients with replacement; each patient multiplicityweights all their images',
            'invalid_draw':'no positive or negative weighted image -> reject/redraw and count',
            'intervals':'percentile95% for prespecified differences; additionally97.5% individual intervals for two primary AP contrasts (approximate Bonferroni family95%)',
            'conditional_scope':'three fixed training fits; seed variability separately described; no p-values',
            'checkpoint_every':100,'cpu_threads':2},
        'thresholds':{'source':'each fixed model/seed/head own HAM_val NATIVE', 'targets':[.9,.95],
            'selection':'max specificity subject to target sensitivity; tie maximum threshold',
            'positive':'p>=threshold','transport':'same numeric threshold to HAM_test,SIIM_E_REAL,SIIM_E_NULL',
            'outputs':['TP','FN','FP','TN','sensitivity','specificity','PPV','NPV'], 'CI':False},
        'curves':'complete ROC/PR per run/condition/head; no smoothing or selection',
        'prohibited':['calibration fitting','TTA','ensembling','external threshold tuning','additional training','new cohort','extra localization','manuscript edits'],
        'manifest':manifest}
    target = OUT / 'P3_FROZEN_PROTOCOL.json'
    if target.exists():
        previous = read(target)
        assert {k:v for k,v in previous.items() if k != 'frozen_at'} == protocol, 'Freeze changed'
    else:
        dump(target, {'frozen_at':datetime.now().astimezone().isoformat(), **protocol})
    dump(OUT / 'FIXED_MODEL_IDENTITIES.json', models)
    update_progress(status='EVALUATION_READY', training_status='COMPLETE',training_completed=3,
                    evaluation_status='READY', statistics_status='NOT_STARTED')
    log('ALL_15_BESTS_AND_ANALYSIS_FROZEN')


if __name__ == '__main__':
    freeze()
