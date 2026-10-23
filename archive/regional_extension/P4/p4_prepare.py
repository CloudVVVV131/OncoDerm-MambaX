"""P4 non-performance cohort binding, bounded duplicate checks, and protocol freeze.

No training, model forwarding, external threshold selection, or global scanning.
"""
import csv
import hashlib
import shutil
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
from p4_common import *

POSITIVE = ['melanoma', 'melanoma (in situ)', 'melanoma (less than 0.76 mm)',
            'melanoma (0.76 to 1.5 mm)', 'melanoma (more than 1.5 mm)', 'melanoma metastasis']
NEGATIVE = ['basal cell carcinoma', 'blue nevus', 'clark nevus', 'combined nevus',
            'congenital nevus', 'dermal nevus', 'dermatofibroma', 'lentigo', 'melanosis',
            'miscellaneous', 'recurrent nevus', 'reed or spitz nevus', 'seborrheic keratosis', 'vascular lesion']
LOCATION = {'abdomen':'abdomen', 'acral':'acral', 'back':'back', 'chest':'chest',
            'lower limbs':'lower extremity', 'upper limbs':'upper extremity',
            'genital areas':'genital', 'head neck':'unknown', 'buttocks':'unknown'}

def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def save_csv(path, rows, fields=None):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader(); w.writerows(rows)

def token(value):
    return hashlib.sha256(('Derm7pt_P4|' + str(value)).encode()).hexdigest()

def prepare():
    assert not (OUT/'P4_FROZEN_PROTOCOL.json').exists(), 'Already frozen; do not rebuild after predictions'
    assert shutil.disk_usage(OUT).free >= 2 * 1024**3
    PRIVATE.mkdir(exist_ok=True)
    update_progress(status='PREPARING',evaluation_status='NOT_STARTED',statistics_status='NOT_STARTED')
    confirmation = {
        'recorded_at':now(), 'source':'explicit user confirmation in current conversation, 2026-10-16',
        'scope':'Current Frontiers M0, R_IMG, R_CLIN, XATTN, B2S and their development process only',
        'statement':'Derm7pt was not used for training, auxiliary training, development/model design selection, hyperparameter tuning, checkpoint selection, or method changes based on these models performance on Derm7pt.',
        'basis':'User identifies current training, selection and evaluation records, NOT download date',
        'other_research_lines':'not covered, not inspected',
        'independence_limit':'This use-history attestation does not prove absence of cross-dataset patient or lesion overlap.'}
    dump(OUT/'USAGE_HISTORY_CONFIRMATION.json',confirmation)
    previous=read(P3/'P3_FROZEN_PROTOCOL.json')
    evidence=[]
    def source(path, expected=None):
        path=Path(path); digest=sha(path)
        if expected is not None: assert digest == expected, str(path)
        evidence.append({'path':str(path),'sha256':digest,'bytes':path.stat().st_size})
        return digest
    source(P3/'P3_FROZEN_PROTOCOL.json')
    source(DATA_PARENT/'DOWNLOAD_REPORT.md')
    inv_path=DATA_PARENT/'verification_20261015_234206_685503/EXTRACTED_FILE_INVENTORY.csv'
    source(inv_path)
    old_inventory={r['archive_path']:r for r in table(inv_path)}
    meta_hash=source(DATA/'meta/meta.csv',old_inventory['release_v0/meta/meta.csv']['sha256'])
    rows=table(DATA/'meta/meta.csv')
    assert len(rows)==1011 and len({r['case_num'] for r in rows})==1011
    assert set(r['diagnosis'] for r in rows)==set(POSITIVE+NEGATIVE)
    assert set(r['sex'] for r in rows)=={'female','male'}
    assert set(r['location'] for r in rows)==set(LOCATION)
    assert sum(r['diagnosis'] in POSITIVE for r in rows)==252
    rows=sorted(rows,key=lambda r:int(r['case_num']))
    assert len({r['derm'] for r in rows})==1011
    mapping=read(SOURCE/'metadata_mapping.json')
    source(SOURCE/'metadata_mapping.json')
    assert mapping['dimension']==22 and len(mapping['token_order'])==22
    sex_vocab=mapping['sex']['vocabulary']; site_vocab=mapping['site']['vocabulary']
    dump(OUT/'CLINICAL_MAPPING.json',{
        'token_order':mapping['token_order'],'age_value':0,'age_missing':1,
        'sex_vocabulary':sex_vocab,'site_vocabulary':site_vocab,
        'location_mapping':LOCATION,'unsupported_nonempty':['head neck','buttocks'],
        'unsupported_nonempty_site_missing':0,'E_NULL':mapping['E_NULL'],
        'E_AVAILABLE':'Observed sex and location only; age is missing for every case',
        'forbidden_model_fields':['diagnosis','seven_point_score','seven_point_attributes','elevation','management','difficulty','case_num','case_id','notes','path','ground_truth_mask']})
    source_records=read(FIRST/'INPUT_SOURCES.json')
    ham_path=ROOT/'project/revision/manifests/HAM10000_recovered_images.csv'
    ham_ref=next(r for r in source_records if r['path']==str(ham_path))
    source(ham_path,ham_ref['sha256'])
    ham=table(ham_path)
    assert len(ham)==10015
    split_by_id={}
    for split in ['train','val','test']:
        item=next(r for r in previous['manifest'] if r['path'].endswith(f'ham10000_seed42_{split}.csv'))
        source(item['path'],item['sha256'])
        for r in table(item['path']):
            assert r['image_id'] not in split_by_id
            split_by_id[r['image_id']]=split
    assert set(split_by_id)=={r['image_id'] for r in ham}
    aux_path=FIRST/'fim_mask_train100.csv'
    source(aux_path,read(FIRST/'DATA_CHECK.json')['selected_manifest_sha256'])
    aux=table(aux_path); assert len(aux)==100
    comparators=[]
    for r in ham:
        comparators.append({'id':r['image_id'],'file':r['sha256'],'pixel':r['pixel_sha256'],
                            'source':'HAM_'+split_by_id[r['image_id']]})
    for r in aux:
        comparators.append({'id':r['image_id'],'file':r['image_sha256'],'pixel':r['pixel_sha256'],'source':'fixed_aux100'})
    index={key:defaultdict(list) for key in ['id','file','pixel']}
    for r in comparators:
        for key in index: index[key][r[key]].append(r)
    masks=[]; private=[]; matches=[]; fingerprints=[]; public_fingerprints=[]
    for i,r in enumerate(rows):
        relative=r['derm']
        # Exact, original-case lookup only. No image indexing or case normalization.
        assert relative and '..' not in Path(relative).parts and not Path(relative).is_absolute()
        path=DATA/'images'/relative
        prior=old_inventory['release_v0/images/'+relative]
        digest=sha(path); assert digest==prior['sha256'] and path.stat().st_size==int(prior['bytes'])
        with Image.open(path) as im:
            rgb=im.convert('RGB'); size=rgb.size
            pixels=hashlib.sha256(rgb.tobytes()).hexdigest()
        matched=[]
        for kind,value in [('id',Path(relative).stem),('file',digest),('pixel',pixels)]:
            for other in index[kind].get(value,[]):
                matched.append({'case_num':r['case_num'],'match_kind':kind,'matched_id':other['id'],'matched_cohort':other['source']})
        matches.extend(matched)
        # Exclude definite train/development/auxiliary image identity. HAM test matches are reported separately.
        reasons=sorted({m['matched_cohort']+':'+m['match_kind'] for m in matched
                        if m['matched_cohort'] in ['HAM_train','HAM_val','fixed_aux100']})
        include=not reasons
        vector=[0.] + [float(r['sex']==s) for s in sex_vocab] + [float(LOCATION[r['location']]==s) for s in site_vocab] + [1.,0.,0.]
        assert len(vector)==22 and vector[19:]==[1,0,0]
        masks.append(vector)
        private.append({'source_order':i,'case_num':r['case_num'],'derm':relative,
            'target':int(r['diagnosis'] in POSITIVE),'sex':r['sex'],'location':r['location'],
            'clinic_equals_derm':r['clinic']==relative,'included':include,'exclusion_reasons':';'.join(reasons),
            'file_sha256':digest,'rgb_pixel_sha256':pixels,'width':size[0],'height':size[1]})
        fingerprints.append((digest,pixels,relative))
        public_fingerprints.append({'case_token':token(r['case_num']),'derm_path_sha256':hashlib.sha256(relative.encode()).hexdigest(),
             'file_sha256':digest,'rgb_pixel_sha256':pixels,'included':include,'exclusion_reasons':';'.join(reasons)})
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    seen={}
    for i,fp in enumerate(fingerprints):
        for kind,value in enumerate(fp):
            key=(kind,value)
            if key in seen:
                a,b=find(i),find(seen[key]); parent[max(a,b)]=min(a,b)
            else: seen[key]=i
    kept=[i for i,r in enumerate(private) if r['included']]
    for i,r in enumerate(private):
        r['cluster_case_num']=rows[find(i)]['case_num']
        public_fingerprints[i]['cluster_token']=token(r['cluster_case_num'])
    aligned=[private[i] for i in kept]
    groups=sorted({int(r['cluster_case_num']) for r in aligned})
    group_index={v:i for i,v in enumerate(groups)}
    cluster=np.array([group_index[int(r['cluster_case_num'])] for r in aligned],dtype=np.int64)
    labels=np.array([r['target'] for r in aligned],dtype=np.int64)
    assert set(labels)=={0,1}
    np.save(PRIVATE/'E_AVAILABLE.npy',np.asarray(masks,dtype=np.float32)[kept],allow_pickle=False)
    np.save(PRIVATE/'labels.npy',labels,allow_pickle=False)
    np.save(PRIVATE/'clusters.npy',cluster,allow_pickle=False)
    save_csv(PRIVATE/'COHORT_ALIGNMENT.csv',aligned)
    save_csv(PRIVATE/'ALL_CANDIDATES.csv',private)
    save_csv(PRIVATE/'DUPLICATE_MATCHES.csv',matches,['case_num','match_kind','matched_id','matched_cohort'])
    save_csv(OUT/'COHORT_FINGERPRINTS.csv',public_fingerprints)
    counts=Counter(r['diagnosis'] for r in rows)
    included_counts=Counter(rows[i]['diagnosis'] for i in kept)
    save_csv(OUT/'DIAGNOSIS_DISTRIBUTION.csv',[{'diagnosis':d,'binary_label':'melanoma' if d in POSITIVE else 'non-melanoma',
              'before_exclusion':counts[d],'after_exclusion':included_counts[d]} for d in sorted(counts)])
    save_csv(OUT/'CLINICAL_FIELD_COUNTS.csv',[
        {'field':field,'value':value,'before_exclusion':n,'after_exclusion':sum(rows[i][field]==value for i in kept)}
        for field in ['sex','location'] for value,n in sorted(Counter(r[field] for r in rows).items())])
    binding={'candidate_rows':1011,'candidate_positive':252,'candidate_negative':759,
        'included_rows':len(kept),'included_positive':int(labels.sum()),'included_negative':int((1-labels).sum()),
        'prevalence':float(labels.mean()),'excluded_rows':1011-len(kept),
        'reference_scope':{'HAM_train':6982,'HAM_val':1513,'HAM_test_report_only':1520,'fixed_aux100':100},
        'match_counts':dict(Counter(m['matched_cohort']+':'+m['match_kind'] for m in matches)),
        'exclusion_counts':dict(Counter(reason for r in private for reason in r['exclusion_reasons'].split(';') if reason)),
        'clinic_derm_same_row_reference_count':sum(r['clinic_equals_derm'] for r in private),
        'derm_only':True,'case_num_unique':True,'case_id_used_as_patient':False,
        'resampling_unit':'case_num, connected components for confirmed same-path/file/RGB-pixel identity',
        'clusters':len(groups),'multi_case_clusters':sum(n>1 for n in Counter(cluster).values()),
        'patient_level_claim':False,'cluster_order':'ascending numeric representative case_num',
        'limitations':['No reliable patient or cross-case lesion linkage is supplied for this evaluation.',
            'Exact original-file and unresized RGB-pixel hashes plus exact named IDs checked against bounded current-direction manifests only.',
            'No new perceptual/near-duplicate search; no claim that recompressed/cropped near duplicates or all cross-dataset patients are excluded.',
            'HAM test-only matches are disclosed but not treated as training/development exclusions.',
            'non-melanoma includes malignant non-melanoma diagnoses and does not mean all benign.'],
        'meta_sha256':meta_hash,'alignment_sha256':sha(PRIVATE/'COHORT_ALIGNMENT.csv'),
        'clinical_tensor_sha256':sha(PRIVATE/'E_AVAILABLE.npy'),'labels_sha256':sha(PRIVATE/'labels.npy'),
        'cluster_tensor_sha256':sha(PRIVATE/'clusters.npy')}
    dump(OUT/'COHORT_BINDING.json',binding)
    identities=previous['models']; assert len(identities)==15
    assert {(m['model'],m['training_seed']) for m in identities}=={(m,s) for m in MODELS for s in SEEDS}
    for model in identities:
        assert Path(model['path']).stat().st_size==model['bytes']
        source(model['path'],model['sha256'])
        source(model['config_snapshot'],model['effective_config_sha256'])
        cfg=read(model['config_snapshot'])['original_config']
        assert cfg['data']['metadata']['sex_values']==sex_vocab
        assert cfg['data']['metadata']['localization_values']==site_vocab
        assert cfg['data']['input_size']==224 and cfg['data']['resize_short_side']==256
    thresholds=P3/'DEVELOPMENT_THRESHOLDS.csv'
    source(thresholds)
    shutil.copy2(thresholds,OUT/'DEVELOPMENT_THRESHOLDS.csv')
    assert len(table(thresholds))==60
    dump(OUT/'FIXED_MODEL_IDENTITIES.json',identities)
    for item in previous['manifest']:
        old=Path(item['path'])
        if old.is_relative_to(P3/'source'):
            copied=SOURCE/old.relative_to(P3/'source')
            source(copied,item['sha256'])
    for path in sorted(OUT.glob('*.py')): source(path)
    for name in ['USAGE_HISTORY_CONFIRMATION.json','CLINICAL_MAPPING.json','COHORT_BINDING.json','COHORT_FINGERPRINTS.csv',
                 'DIAGNOSIS_DISTRIBUTION.csv','CLINICAL_FIELD_COUNTS.csv','DEVELOPMENT_THRESHOLDS.csv']:
        source(OUT/name)
    for name in ['COHORT_ALIGNMENT.csv','E_AVAILABLE.npy','labels.npy','clusters.npy']: source(PRIVATE/name)
    protocol={'frozen_at':now(),'started_at':START.isoformat(),'deadline':datetime.fromtimestamp(DEADLINE,ZONE).isoformat(),
        'scope':'P4 fixed external evaluation only; no training or prior-stage rerun',
        'usage_history':confirmation,'cohort':binding,'data_root':str(DATA),'models':identities,'new_training_runs':0,
        'labels':{'positive':POSITIVE,'negative':NEGATIVE,'multiclass_HAM_remapping':False},
        'clinical_mapping':read(OUT/'CLINICAL_MAPPING.json'),'conditions':['E_AVAILABLE','E_NULL'],
        'heads':['softmax_MEL','endpoint_MEL'],'primary_endpoint':'Derm7pt E_AVAILABLE softmax_MEL AP',
        'primary_comparisons':['R_CLIN_minus_M0','R_CLIN_minus_XATTN'],
        'supporting_comparisons':['R_CLIN_minus_R_IMG','R_CLIN_minus_B2S'],
        'interval_scope':'four contrasts x two conditions x two heads x AP/AUC; two primary AP contrasts also 97.5%',
        'preprocessing':previous['preprocessing'],'precision':previous['precision'],'inference_batch':8,'workers':2,
        'cache':previous['cache'],'metrics':{k:previous['metrics'][k] for k in ['AP','AUC','Brier','binary_NLL','binary_ECE15']},
        'aggregation':previous['aggregation'],
        'bootstrap':{'draws':10000,'rng':'PCG64','seed':20261017,'clusters':len(groups),
            'unit':binding['resampling_unit'],'order':binding['cluster_order'],'paired_across':'all models/seeds/conditions/heads',
            'single_class_draws':'reject, redraw, and count','intervals':'percentile95%; primary AP additionally97.5% approximate Bonferroni for two comparisons',
            'scope':'case/same-image-cluster sampling conditional on the three fixed training fits; NOT verified patient-level',
            'checkpoint_every':100},
        'thresholds':{'source':str(thresholds),'sha256':sha(thresholds),'targets':[.9,.95],
            'source_cohort':'each original fixed model/seed/head HAM validation NATIVE','comparison':'p>=threshold',
            'external_reselection':False,'calibration_fitting':False},
        'curves':'full unmodified ROC/PR coordinates per run/condition/head',
        'prohibited':['training','fine-tuning','best reselection','external threshold tuning','calibration fitting','ensembling','other research lines','cloud','manuscript edits'],
        'manifest':evidence}
    dump(OUT/'P4_FROZEN_PROTOCOL.json',protocol)
    update_progress(status='FROZEN',evaluation_status='PENDING',statistics_status='PENDING',cohort_rows=len(kept),excluded_rows=1011-len(kept))
    log(f'FROZEN cases={len(kept)} positive={labels.sum()} excluded={1011-len(kept)} clusters={len(groups)}')

if __name__=='__main__':
    prepare()
