"""Bounded identity and cohort binding; no inference or performance computation."""
import ast
import shutil
from collections import Counter, defaultdict
import numpy as np
import yaml
from core_common import *

ALLOWLIST = [
 ('B2',42,'oncoderm_mambax_b2_metadata_cmgf_seed42_20260706_234933',360781827,'7d3b060e46add18b7df02014e1df9869133ddca2dfd8d5fafb36feeb725bd16b'),
 ('B2',2024,'oncoderm_mambax_b2_metadata_cmgf_seed2024_20260708_160522',360781891,'bd9300f6fed9f9cb71796cf25d8bc5468a1003a169be137d7a95e96bd8c4181a'),
 ('B2',3407,'oncoderm_mambax_b2_metadata_cmgf_seed3407_20260708_230453',360781891,'c9173f33ff787eb5533a7abfb0987dd6577618a15f6f51fff847d5d06beaabf5'),
 ('B6',42,'oncoderm_mambax_b6_full_seed42_20260707_004712',401499163,'525302d30405b75229c428d5b1a8d1f282de0e0d25ac98ef13c84fd91b5f923c'),
 ('B6',2024,'oncoderm_mambax_b6_full_seed2024_20260708_163051',401499163,'4772ae5d3ef772c58438ce35915326f10a01d8fcd416999e6f274ecfd005abfa'),
 ('B6',3407,'oncoderm_mambax_b6_full_seed3407_20260708_232735',401499163,'d27998a562ccafeed44bf731674dec95943b3276db80bd1239af645a8c006ef9')]

def main():
    assert not (OUT/'ORIGINALCORE_PROTOCOL.json').exists(), 'Already frozen'
    PRIVATE.mkdir(exist_ok=True)
    progress(status='PREPARING')
    free=shutil.disk_usage(OUT).free
    assert free >= 1024**3, f'Need 1 GiB reserve, free={free}'
    evidence=[]
    def source(path, expected=None):
        path=Path(path)
        digest=sha(path)
        if expected is not None: assert digest==expected, f'SHA mismatch: {path}'
        evidence.append(dict(path=str(path),sha256=digest,bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns))
        return digest
    task=OUT.parent/'FIM_OriginalCore_Derm7pt_Execution.txt'
    source(task)
    shutil.copy2(task,OUT/task.name)
    confirmation=dict(recorded_at=now(),source='Explicit user reply in current conversation on 2026-10-17',
        scope='Six original manuscript B2/B6 runs, training seeds42/2024/3407, and their development process',
        user_statement='确认六个原模型及其开发过程均未主动使用 Derm7pt',
        question_scope=['training','auxiliary training','development/model selection','hyperparameter tuning','checkpoint selection'],
        not_a_guarantee_of_unknown_cross_dataset_overlap=True)
    dump(OUT/'USAGE_HISTORY_CONFIRMATION.json',confirmation)
    manifest={r['path']:r for r in table(B2/'INPUT_MANIFEST.csv')}
    for name in ['INPUT_MANIFEST.csv','B2_FROZEN_PROTOCOL.json','MODEL_SOURCE_MANIFEST.csv','FINAL_QC.json','metadata_mapping.json','PREDICTION_MANIFEST.json']:
        source(B2/name)
    assert read(B2/'FINAL_QC.json')['status']=='PASS'
    assert read(B2/'FINAL_QC.json')['checkpoints_final_SHA_rechecked']==6
    p4=read(P4/'P4_FROZEN_PROTOCOL.json')
    p4manifest={r['path']:r for r in p4['manifest']}
    for name in ['P4_FROZEN_PROTOCOL.json','COHORT_BINDING.json','CLINICAL_MAPPING.json']:
        source(P4/name)
    mapping=read(P4/'CLINICAL_MAPPING.json')
    original_mapping=read(B2/'metadata_mapping.json')
    assert mapping['token_order']==original_mapping['token_order']
    assert mapping['sex_vocabulary']==original_mapping['sex']['vocabulary']
    assert mapping['site_vocabulary']==original_mapping['site']['vocabulary']
    assert mapping['E_NULL']==original_mapping['E_NULL']
    shutil.copy2(P4/'CLINICAL_MAPPING.json',OUT/'CLINICAL_MAPPING.json')
    model_evidence=[]
    for r in table(B2/'MODEL_SOURCE_MANIFEST.csv'):
        old=Path(r['copy_path']); source(old,r['sha256'])
        dest=OUT/'model_code'/old.relative_to(B2/'model_code')
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(old,dest)
        assert sha(dest)==r['sha256']
        model_evidence.append(dict(original_source=r['source_path'],B2_copy=str(old),actual_import_copy=str(dest),sha256=r['sha256'],change='byte-identical to B2 executed copy'))
    csv_save(OUT/'MODEL_SOURCE_MANIFEST.csv',model_evidence)
    identities=[]
    pred_index=read(B2/'PREDICTION_MANIFEST.json')
    parent=ROOT/'oncoderm_stability_v11_full_results_20260709_012310/outputs/runs'
    for family,seed,run,nbytes,digest in ALLOWLIST:
        path=parent/run/'checkpoints/best_val_macro_f1.pth'
        cfgpath=parent/run/'config.yaml'
        assert path.stat().st_size==nbytes and manifest[str(path)]['sha256']==digest
        source(path,digest)
        cfghash=source(cfgpath,manifest[str(cfgpath)]['sha256'])
        cfg=yaml.safe_load(cfgpath.read_text(encoding='utf-8-sig'))
        assert cfg['project']['seed']==seed
        assert cfg['data']['metadata']['sex_values']==mapping['sex_vocabulary']
        assert cfg['data']['metadata']['localization_values']==mapping['site_vocabulary']
        assert cfg['data']['metadata']['age_scale']==100
        assert cfg['data']['input_size']==224 and cfg['data']['resize_short_side']==256
        techpath=B2/f'technical/{family}_seed{seed}.json'
        source(techpath); tech=read(techpath)
        assert tech['run_id']==run and tech['checkpoint_sha256']==digest and tech['strict']
        matches=[r for r in pred_index if r['model']==family and r['training_seed']==seed and r['dataset']=='HAM_val']
        assert len(matches)==1 and matches[0]['run_id']==run and matches[0]['checkpoint_sha256']==digest
        copy=OUT/'original_configs'/f'ORIG_{family}_s{seed}.yaml'
        copy.parent.mkdir(exist_ok=True);shutil.copy2(cfgpath,copy)
        identities.append(dict(model='ORIG_'+family,training_seed=seed,tag=f'ORIG_{family}_s{seed}',run_id=run,
            checkpoint_path=str(path),bytes=nbytes,sha256=digest,config_path=str(cfgpath),config_copy=str(copy),config_sha256=cfghash,
            original_selected_epoch=tech['epoch'],B2_HAM_val_prediction_index=matches[0],technical_source=str(techpath)))
        log('Identity verified '+run)
    dump(OUT/'FIXED_MODEL_IDENTITIES.json',identities)
    binding=read(P4/'COHORT_BINDING.json')
    meta=DATA/'meta/meta.csv'
    source(meta,'471199fb86bcc97cd1521bcc4255bb3f9c563e2ed0ad3ff836392d81a3ed9d3b')
    raw=table(meta);raw_by={int(r['case_num']):r for r in raw}
    assert len(raw)==1011 and len(raw_by)==1011
    labels=p4['labels']; allowed=set(labels['positive']+labels['negative'])
    assert len(allowed)==20 and {r['diagnosis'] for r in raw}==allowed
    assert sum(r['diagnosis'] in labels['positive'] for r in raw)==252
    for name,key in [('COHORT_ALIGNMENT.csv','alignment_sha256'),('E_AVAILABLE.npy','clinical_tensor_sha256'),('labels.npy','labels_sha256'),('clusters.npy','cluster_tensor_sha256')]:
        source(P4/'private_predictions'/name,binding[key])
    rows=table(P4/'private_predictions/COHORT_ALIGNMENT.csv')
    assert len(rows)==1011 and binding['excluded_rows']==0
    assert [int(r['case_num']) for r in rows]==sorted(raw_by)
    ham_path=ROOT/'project/revision/manifests/HAM10000_recovered_images.csv'
    source(ham_path,manifest[str(ham_path)]['sha256'])
    assert manifest[str(ham_path)]['sha256']==p4manifest[str(ham_path)]['sha256']
    ham={r['image_id']:r for r in table(ham_path)}
    index={k:defaultdict(list) for k in ['id','file','pixel']}
    split_checks=[]
    for split,expected_n in [('train',6982),('val',1513)]:
        sp=ROOT/f'oncoderm_bioengineering_full_results_20260817_110459/project/splits/ham10000_seed42_{split}.csv'
        d=source(sp,p4manifest[str(sp)]['sha256'])
        members=table(sp);assert len(members)==expected_n
        if split=='val': assert d==manifest[str(sp)]['sha256']
        for r in members:
            h=ham[r['image_id']]
            for key,value in [('id',r['image_id']),('file',h['sha256']),('pixel',h['pixel_sha256'])]:
                index[key][value].append((split,r['image_id']))
        split_checks.append(dict(split=split,path=str(sp),sha256=d,members=len(members),P4_same_sha256=True,
                                 origin='original manuscript split42; B2 original validation reference and configs'))
    kept=[]; exclusions=[];fingerprints=[]
    for i,r in enumerate(rows):
        src=raw_by[int(r['case_num'])]
        assert r['derm']==src['derm'] and r['sex']==src['sex'] and r['location']==src['location']
        assert int(r['target'])==int(src['diagnosis'] in labels['positive'])
        path=DATA/'images'/r['derm']
        assert path.is_file() and sha(path)==r['file_sha256'], str(path)
        hits=[]
        for kind,value in [('id',Path(r['derm']).stem),('file',r['file_sha256']),('pixel',r['rgb_pixel_sha256'])]:
            hits.extend(f'HAM_{s}:{kind}:{iid}' for s,iid in index[kind].get(value,[]))
        if hits: exclusions.append(dict(case_num=r['case_num'],reason=';'.join(sorted(hits))))
        else: kept.append(i)
        fingerprints.append(dict(case_token=hashlib.sha256(('OriginalCore|'+r['case_num']).encode()).hexdigest(),
             file_sha256=r['file_sha256'],rgb_pixel_sha256=r['rgb_pixel_sha256'],included=not hits,
             exclusion_reason=';'.join(sorted(hits))))
    chosen=[rows[i] for i in kept]
    csv_save(PRIVATE/'COHORT_ALIGNMENT.csv',chosen)
    csv_save(PRIVATE/'EXCLUSIONS.csv',exclusions,['case_num','reason'])
    for name in ['E_AVAILABLE.npy','labels.npy']:
        np.save(PRIVATE/name,np.load(P4/'private_predictions'/name,allow_pickle=False)[kept],allow_pickle=False)
    group_ids=sorted({int(r['cluster_case_num']) for r in chosen})
    cluster=np.array([group_ids.index(int(r['cluster_case_num'])) for r in chosen],dtype=np.int64)
    np.save(PRIVATE/'clusters.npy',cluster,allow_pickle=False)
    y=np.load(PRIVATE/'labels.npy',allow_pickle=False)
    assert set(y)=={0,1}
    csv_save(OUT/'COHORT_FINGERPRINTS.csv',fingerprints)
    csv_save(OUT/'DIAGNOSIS_DISTRIBUTION.csv',[dict(diagnosis=d,label='melanoma' if d in labels['positive'] else 'non-melanoma',before_exclusion=sum(r['diagnosis']==d for r in raw),after_exclusion=sum(raw_by[int(r['case_num'])]['diagnosis']==d for r in chosen)) for d in sorted(allowed)])
    binding_new=dict(candidate_rows=len(raw),included_rows=len(chosen),positive=int(y.sum()),negative=int((1-y).sum()),prevalence=float(y.mean()),excluded_rows=len(exclusions),
       clusters=len(group_ids),resampling_unit='case_num / confirmed same-image clusters; NOT verified patients',cluster_order='numeric representative case_num ascending',
       clinic_derm_same_reference_count=sum(raw_by[int(r['case_num'])]['clinic']==r['derm'] for r in chosen),
       HAM_train_val_checks=split_checks,exact_duplicate_matches=len(exclusions),
       coverage='Exact named IDs and prior file/RGB-pixel hashes for original HAM train/val. Current derm file SHA rechecked; P4 same-image clusters and pixel fingerprints reused.',
       original_B6_auxiliary100='UNKNOWN membership; not traced or reconstructed; no claim of verified nonoverlap. New FIM auxiliary100 is NOT a substitute.',
       limitations=['No verified patient/cross-case lesion linkage.','No near-duplicate/perceptual search; unknown cross-dataset relationships may remain.','non-melanoma is not synonymous with benign.'])
    for name,key in [('COHORT_ALIGNMENT.csv','alignment_sha256'),('E_AVAILABLE.npy','clinical_sha256'),('labels.npy','labels_sha256'),('clusters.npy','clusters_sha256')]: binding_new[key]=sha(PRIVATE/name)
    dump(OUT/'COHORT_BINDING.json',binding_new)
    threshold_path=B2/'DEVELOPMENT_THRESHOLDS.csv'
    thresholds_status='MISSING'
    threshold_info=dict(source=str(threshold_path),status=thresholds_status)
    if threshold_path.is_file():
        th=table(threshold_path)
        expected={(m['model'][5:],str(m['training_seed']),h,str(t)) for m in identities for h in HEADS for t in [.9,.95]}
        actual={(r['model'],r['training_seed'],r['score'],r['target_sensitivity']) for r in th}
        if len(th)==24 and actual==expected and all(r['source']=='HAM_val_NATIVE' for r in th):
            digest=source(threshold_path)
            shutil.copy2(threshold_path,OUT/'DEVELOPMENT_THRESHOLDS_ORIGINAL.csv')
            joined=[]
            for r in th:
                m=next(m for m in identities if m['model']=='ORIG_'+r['model'] and m['training_seed']==int(r['training_seed']))
                joined.append(dict(**r,run_id=m['run_id'],checkpoint_sha256=m['sha256'],prediction_identity_source='B2 PREDICTION_MANIFEST HAM_val record'))
            csv_save(OUT/'THRESHOLD_IDENTITY_JOIN.csv',joined)
            thresholds_status='SOURCE_VERIFIED'
            threshold_info=dict(source=str(threshold_path),sha256=digest,status=thresholds_status,source_rows=24,
                                run_id_binding='B2 PREDICTION_MANIFEST validated unique family/seed -> full run_id and checkpoint SHA; not inferred from generic names')
    metric_source=P4/'p4_metric_core.py';source(metric_source)
    tree=ast.parse(metric_source.read_text(encoding='utf-8-sig'))
    pure=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['GroupedMetric','ece','binary']]
    assert len(pure)==3
    (OUT/'core_metrics.py').write_text('"""AST-extracted pure arithmetic from P4; no P4 imports or drivers."""\nimport numpy as np\nfrom sklearn.metrics import roc_auc_score, average_precision_score\n\n'+ '\n\n'.join(ast.unparse(n) for n in pure)+'\n',encoding='utf-8')
    source(B2/'B2_METRICS.csv')
    siim=[r for r in table(B2/'B2_METRICS.csv') if r['dataset']=='SIIM']
    assert len(siim)==24
    csv_save(OUT/'SIIM_B2_AGGREGATES_SOURCE.csv',siim)
    protocol=dict(frozen_at=now(),started_at=START.isoformat(),deadline=datetime.fromtimestamp(DEADLINE,ZONE).isoformat(),
       status='FROZEN_BEFORE_INFERENCE',purpose='Additional original-core evaluation after P1-P6/P4 review, NOT originally preregistered confirmation; P3/P4 comparisons unchanged',
       new_training_runs=0,usage_history=confirmation,models=identities,cohort=binding_new,clinical_mapping=mapping,labels=labels,
       conditions=CONDITIONS,heads=HEADS,preprocessing='RGB; shorter side256 BILINEAR antialias=True; center224; ImageNet mean/std',
       precision='FP32 eval/inference_mode, TF32 off',batch=8,workers=2,cache_equivalence=dict(atol=1e-5,rtol=1e-4,max_images=8,fallback='full forward'),
       metrics=['average_precision','roc_auc','unclipped_brier','binary_nll_eps1e-7','ece15_equal_width_last_includes1'],
       aggregation='Per-run metric -> mean and sample SD ddof1; no pooled probabilities/ensembles',
       fixed_model_intervals='E_AVAILABLE prob_mel family mean AP/AUC 95%',
       input_effects=['ORIG_B2 E_AVAILABLE minus E_NULL','ORIG_B6 E_AVAILABLE minus E_NULL'],
       input_effect_intervals='both heads AP/AUC 95%; only two prob_mel AP effects also97.5%, approximate Bonferroni reference, still additional analysis',
       config_comparison='ORIG_B6 minus ORIG_B2; both conditions/heads AP/AUC95% supporting only, NOT single-factor ablation',
       bootstrap=dict(valid_draws=10000,rng='PCG64',seed=20261021,quantile_method='linear',cluster_order='numeric representative case_num ascending',
          single_class='redraw and count',same_sample_all=True,resample_training_seed=False,conditional_on_fixed_models=True,checkpoint_every=100),
       thresholds=threshold_info,threshold_application='Original value, p>=t, both inputs; no reselection or calibration fit',
       curves='24 full unsmoothed ROC/PR score vectors',cross_cohort='B2 executed FP32 SIIM aggregates and current FP32 Derm7pt; no AP subtraction across prevalence/age coverage',
       disk_free_at_preparation=free,estimated_new_disk_reserve=1024**3,evidence=evidence)
    dump(OUT/'ORIGINALCORE_PROTOCOL.json',protocol)
    progress(status='FROZEN',identity='HASHES_VERIFIED_STRICT_LOAD_PENDING',thresholds=thresholds_status,cohort_rows=len(chosen))
    log(f'FROZEN models=6 derm_cases={len(chosen)} positives={y.sum()} excluded={len(exclusions)} thresholds={thresholds_status}')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        progress(status='BLOCKED',error=repr(exc));raise
