"""B2 input/hash/mapping preparation. No model loading, inference or training."""
from b2_common import *
import ast
import copy
import shutil
from collections import Counter
import numpy as np
import pandas as pd
import yaml

INPUTS=[]
SEX=['male','female','unknown']
SITE=['abdomen','acral','back','chest','ear','face','foot','genital','hand','lower extremity','neck','scalp','trunk','unknown','upper extremity']
NULL=[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1]


def observe(path,role,expected=''):
    p=Path(path)
    assert p.is_file(), 'Missing exact input: '+str(p)
    s=p.stat(); h=sha(p)
    assert not expected or h==expected,'SHA mismatch: '+str(p)
    INPUTS.append(dict(path=str(p),role=role,bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=h,expected_sha256=expected,expected_match=bool(not expected or h==expected)))
    return h


def normalize(v):
    if pd.isna(v): return 'unknown'
    value=' '.join(str(v).strip().lower().split())
    return value or 'unknown'


def code_copy():
    # Import only copied model/data modules; do not run the historical registry or drivers.
    paths=['lesionmamba/__init__.py','lesionmamba/data/__init__.py','lesionmamba/data/datasets.py',
           'lesionmamba/data/transforms.py','lesionmamba/models/oncoderm_mambax.py']
    records=[]
    for rel in paths:
        src=ROOT/'project'/rel
        h=observe(src,'model_source_copy')
        dest=OUT/'model_code'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        assert not dest.exists()
        shutil.copy2(src,dest)
        assert sha(dest)==h
        records.append(dict(source_path=str(src),copy_path=str(dest),sha256=h,change='byte_identical'))
    init=OUT/'model_code/lesionmamba/models/__init__.py'
    with init.open('x',encoding='ascii') as f:
        f.write('"""B2 isolated model namespace; no historical driver imports."""\n')
    records.append(dict(source_path='new isolation-only package initializer',copy_path=str(init),sha256=sha(init),change='Empty model initializer bypasses unrelated registry imports; model implementation unchanged'))
    enc=js(NAV/'B2_ENCODER_NAVIGATION.json')
    observe(enc['path'],'historical_encoder_reference',enc['sha256'])
    old=ast.parse(Path(enc['path']).read_text(encoding='utf-8'))
    new=ast.parse((OUT/'model_code/lesionmamba/data/datasets.py').read_text(encoding='utf-8'))
    for name in ['metadata_dim','_norm_text','encode_metadata']:
        find=lambda tree:ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        assert find(old)==find(new), 'Encoder function differs: '+name
    save_csv('MODEL_SOURCE_MANIFEST.csv',records)


def main():
    LOCAL.mkdir(exist_ok=True)
    for d in ['cache','predictions','metadata','technical']:
        (LOCAL/d).mkdir(exist_ok=True)
    expected_start=js(OUT/'EXTRACTION_RECEIPT.json')['source_sha256']
    observe(ROOT/'OncoDerm_B2_Inference_Only_Start.zip','start_zip',expected_start)
    package_checks=[]
    for row in csv_rows(PACKAGE/'PACKAGE_MANIFEST.csv'):
        p=PACKAGE/row['relative_path']; h=sha(p)
        assert h==row['sha256'] and p.stat().st_size==int(row['bytes'])
        package_checks.append(dict(path=str(p),sha256=h,bytes=p.stat().st_size,match=True))
    save_json('PACKAGE_INTEGRITY.json',package_checks)
    configs=[]
    for entry in entries():
        tag=model_tag(entry)
        log('Hashing fixed historical checkpoint '+tag)
        observe(entry['checkpoint_path'],'checkpoint',entry['historical_checkpoint_sha256_NOT_recomputed'])
        observe(entry['config_path'],'config',entry['config_sha256_now'])
        cfg=yaml.safe_load(Path(entry['config_path']).read_text(encoding='utf-8-sig'))
        md=cfg['data']['metadata']
        assert md['sex_values']==SEX and md['localization_values']==SITE
        assert md['enabled'] and md['age_scale']==100 and md['include_missing_indicators']
        assert cfg['data']['input_size']==224 and cfg['data']['resize_short_side']==256
        configs.append(cfg)
    code_copy()
    sys.path.insert(0,str(OUT/'model_code'))
    from lesionmamba.data.datasets import encode_metadata,LABEL_MAP
    assert LABEL_MAP['mel']==4 and len(LABEL_MAP)==7
    actual_null=encode_metadata(pd.Series({'age':np.nan,'sex':'unknown','localization':'unknown'}),configs[0]['data']['metadata']).numpy()
    assert np.array_equal(actual_null,np.array(NULL,np.float32))

    mapping_path=ROOT/'_figure_source_freeze_20260819/ISIC_2020_Training_GroundTruth_v2.csv'
    oldinputs={r['path']:r for r in csv_rows(NAV/'INPUT_MANIFEST.csv')}
    observe(mapping_path,'SIIM_original_mapping',oldinputs[str(mapping_path)]['sha256'])
    siim=pd.read_csv(mapping_path,usecols=['image_name','patient_id','target','age_approx','sex','anatom_site_general_challenge'],dtype={'image_name':str,'patient_id':str})
    assert len(siim)==33126 and siim.image_name.is_unique and siim.patient_id.notna().all()
    assert siim.patient_id.nunique()==2056 and int(siim.target.sum())==584
    siim=siim.sort_values('image_name',kind='stable').reset_index(drop=True)
    cov=[]; tokens=[]
    mapped=pd.DataFrame(index=siim.index)
    age=pd.to_numeric(siim.age_approx,errors='coerce')
    valid_age=np.isfinite(age)
    mapped['age']=age.where(valid_age,np.nan)
    cov.append(dict(field='age_approx',status='finite',count=int(valid_age.sum())))
    cov.append(dict(field='age_approx',status='missing_or_unparseable_nonfinite',count=int((~valid_age).sum())))
    cov.append(dict(field='age_approx',status='finite_outside_0_120_report_only_not_clipped',count=int((valid_age&((age<0)|(age>120))).sum())))
    for source,target,vocab in [('sex','sex',SEX),('anatom_site_general_challenge','localization',SITE)]:
        source_values=siim[source].map(normalize)
        source_values=source_values.astype(str)
        mapped_values=source_values.replace({'torso':'trunk'}) if target=='localization' else source_values
        mapped[target]=mapped_values
        counts=Counter(source_values)
        status_counts=Counter()
        for raw,n in sorted(counts.items()):
            value='trunk' if target=='localization' and raw=='torso' else raw
            status=('raw_missing_or_unknown' if raw=='unknown' else 'predeclared_synonym' if raw!=value else 'direct_vocabulary_match' if value in vocab else 'unsupported_nonempty')
            status_counts[status]+=n
            tokens.append(dict(field=source,source_value_normalized=raw,normalized_value_passed_to_encoder=value,
                encoder_token=value if value in vocab else 'unknown',missing_flag=int(raw=='unknown'),status=status,count=n))
        for status in ['raw_missing_or_unknown','predeclared_synonym','direct_vocabulary_match','unsupported_nonempty']:
            cov.append(dict(field=source,status=status,count=status_counts[status]))
    real=np.stack([encode_metadata(row,configs[0]['data']['metadata']).numpy() for _,row in mapped.iterrows()])
    assert real.shape==(33126,22) and np.isfinite(real).all()
    # All clinical vectors are private local intermediates, never delivery payloads.
    with (LOCAL/'metadata/SIIM_E_REAL.npy').open('xb') as f:np.save(f,real,allow_pickle=False)
    siim[['image_name','patient_id','target']].to_csv(LOCAL/'metadata/SIIM_alignment.csv',index=False,mode='x')
    save_csv('METADATA_COVERAGE.csv',cov)
    save_csv('METADATA_TOKEN_MAPPING.csv',tokens)
    save_json('metadata_mapping.json',dict(frozen_at=now(),cohort='SIIM_ISIC2020',dimension=22,
        age=dict(source='age_approx',destination='age',scale=100,invalid='missing',outliers='reported_not_clipped'),
        sex=dict(source='sex',vocabulary=SEX),site=dict(source='anatom_site_general_challenge',destination='localization',vocabulary=SITE,synonyms={'torso':'trunk'}),
        token_order=['age/100']+['sex_'+x for x in SEX]+['site_'+x for x in SITE]+['age_missing','sex_missing','site_missing'],
        normalization='strip/lower/collapse whitespace; actual missing/empty/unknown -> unknown',
        unsupported_nonempty='unknown one-hot with missing flag 0; preserve normalized input to historical encoder',
        E_NULL=NULL,labels_patient_image_ids_never_model_inputs=True,
        source_hash=sha(mapping_path),private_tensor_sha256=sha(LOCAL/'metadata/SIIM_E_REAL.npy'),mapping_selected_without_predictions=True))

    image_nav=js(NAV/'B2_IMAGE_NAVIGATION.json')
    images={}
    for nav in image_nav:
        p=Path(nav['manifest_path']); observe(p,'recovered_image_manifest',nav['manifest_sha256'])
        frame=pd.read_csv(p,dtype={'image_id':str})
        assert frame.image_id.is_unique
        images[nav['dataset']]=frame.set_index('image_id')
    datasets=[('SIIM',siim.image_name.tolist(),images['SIIM2020'])]
    ham_checks=[]
    for split,n in [('val',1513),('test',1520)]:
        p=ROOT/f'oncoderm_bioengineering_full_results_20260817_110459/project/splits/ham10000_seed42_{split}.csv'
        observe(p,'original_HAM_'+split)
        df=pd.read_csv(p,dtype={'image_id':str,'lesion_id':str})
        assert len(df)==n and df.image_id.is_unique and df.dx.isin(LABEL_MAP).all()
        df=df.sort_values('image_id',kind='stable').reset_index(drop=True)
        # Cross-check identities against every original run prediction, not rebuilt Q data.
        for entry in entries():
            cp=Path(entry['checkpoint_path'])
            prediction=cp.parents[3]/'predictions'/f"{entry['run_id']}_{split}_predictions.csv"
            observe(prediction,'historical_HAM_ID_reference')
            ids=pd.read_csv(prediction,usecols=['image_id'],dtype=str).image_id
            assert ids.is_unique and set(ids)==set(df.image_id)
        df.to_csv(LOCAL/f'metadata/HAM_{split}.csv',index=False,mode='x')
        ham_checks.append(dict(split=split,rows=len(df),source=str(p),sha256=sha(p),six_historical_ID_sets_match=True))
        datasets.append(('HAM_'+split,df.image_id.tolist(),images['HAM10000']))
    a=pd.read_csv(LOCAL/'metadata/HAM_val.csv'); b=pd.read_csv(LOCAL/'metadata/HAM_test.csv')
    assert not set(a.image_id)&set(b.image_id)
    assert not set(a.lesion_id)&set(b.lesion_id)
    image_checks=[]
    for name,ids,index in datasets:
        assert all(x in index.index for x in ids)
        records=[]
        for i in ids:
            row=index.loc[i]; p=Path(row['path'])
            assert p.is_file(), 'Missing image: '+str(p)
            assert p.stat().st_size==int(row['bytes']), 'Image size mismatch: '+str(p)
            records.append(dict(image_id=i,path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256_historical=row['sha256']))
        save_csv('local_only/metadata/'+name+'_images.csv',records)
        image_checks.append(dict(dataset=name,count=len(ids),exists_and_bytes_match=True,decode='Deferred to actual inference; any failure stops unit'))
    save_json('INPUT_VALIDATION.json',dict(checked_at=now(),checkpoints_SHA_verified=6,configs_SHA_verified=6,
        HAM=ham_checks,images=image_checks,SIIM=dict(images=33126,patients=2056,positive_images=584),
        encoder_AST_matches_historical=True,NULL_matches_reference=True,metadata_finite=True))
    save_csv('INPUT_MANIFEST.csv',INPUTS)
    save_json('B2_FROZEN_PROTOCOL.json',dict(frozen_at=now(),task_sha256=sha(PACKAGE/'01_Codex_B2_执行任务.md'),
        scope='B2_only',training=False,cloud=False,precision='FP32_TF32_off_eval_inference_mode',
        models=['B2','B6'],training_seeds=[42,2024,3407],split_seed=42,
        conditions=dict(SIIM=['E_NULL','E_REAL'],HAM_val=['NATIVE'],HAM_test=['NATIVE']),
        scoring=dict(primary='prob_mel=softmax(logits7)[:,4]',secondary='prob_mel_endpoint=sigmoid(mel_logit)'),
        metrics=['AP','ROC_AUC','Brier','binary_NLL','ECE_15_equal_width','unpenalized_logit_calibration_intercept_slope_point_only'],
        probability_clipping=dict(epsilon=1e-7,only_for=['NLL','calibration_logit'],AP_AUC='raw_probabilities'),
        ECE_bins='15 equal width on [0,1]; [lo,hi) except final includes1',
        calibration='Binomial logistic MLE without penalty; evaluation only; failure recorded, no recalibrated probabilities',
        bootstrap=dict(generator='PCG64',seed=20261013,valid_replicates=10000,patient_order='lexicographic exact strings',
            draws_per_replicate=2056,same_resample_all_vectors=True,invalid_one_class='redraw same RNG stream',
            metrics=['AP','ROC_AUC','Brier','fixed_threshold_sensitivity','fixed_threshold_specificity'],
            CI95=[2.5,97.5],primary_common_AP_three_seed_mean_CI975=[1.25,98.75],quantile_method='linear',
            primary_comparisons=['model_B2_mean(E_REAL-E_NULL)_3_fixed_seeds','model_B6_mean(E_REAL-E_NULL)_3_fixed_seeds'],
            conditional_on_fixed_three_runs=True,resample_training_seed=False,refit_calibration=False,reselect_threshold=False,p_values=False),
        thresholds=dict(source='same_run_same_score_HAM_val_NATIVE',target_sensitivity=[.9,.95],
            optimize='maximum specificity among observed-score thresholds meeting sensitivity; tie largest numeric threshold',
            positive='p>=threshold',shared_across_SIIM_conditions=True,test_reselection=False,uncertainty='conditional on chosen validation threshold'),
        technical=dict(start_batch=8,candidates=[8,16,32],selection='memory and time only; not SIIM performance',
            cache_equivalence_atol=1e-5,cache_equivalence_rtol=1e-4,validation_sample='first96 lexicographic original validation image IDs',
            no_cuda='stop',max_total_estimated_seconds=14400,statistical_allowance_seconds=1800),
        input_manifest_sha256=sha(OUT/'INPUT_MANIFEST.csv'),mapping_sha256=sha(OUT/'metadata_mapping.json')))
    save_json('PREPARATION_STATUS.json',dict(status='PASS',time=now(),weights_hash_read=True,weights_loaded=False,inference=False))
    log('Preparation PASS: six weight hashes, mapping, source copies and required image paths verified')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save_json('PREPARATION_BLOCKED.json',dict(status='BLOCKED',time=now(),error=repr(exc)))
        raise
