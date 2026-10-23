"""Derm7pt-only fixed-model statistics; no threshold fitting or old-cohort access."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='2'
import time
import numpy as np
import pandas as pd
from scipy.special import softmax, expit
from sklearn.metrics import roc_curve, precision_recall_curve
from threadpoolctl import threadpool_limits
from p4_common import *
from p4_metric_core import GroupedMetric, confusion, binary, tests

UNITS=[('E_AVAILABLE','softmax_MEL',0,4),('E_AVAILABLE','endpoint_MEL',0,7),
       ('E_NULL','softmax_MEL',1,4),('E_NULL','endpoint_MEL',1,7)]
COMPARATORS=['M0','XATTN','R_IMG','B2S']

def confusion_extended(y,p,t):
    row=confusion(y,p,t)
    row['PPV']=row['TP']/(row['TP']+row['FP']) if row['TP']+row['FP'] else None
    row['NPV']=row['TN']/(row['TN']+row['FN']) if row['TN']+row['FN'] else None
    return row

def complete_prob(identity):
    stem=f"{identity['run']}_Derm7pt"
    state=read(PRIVATE/f'{stem}_status.json')
    assert state['status']=='COMPLETE' and state['next_index']==state['total']
    assert state['best_sha256']==identity['sha256'] and state['protocol_sha256']==sha(OUT/'P4_FROZEN_PROTOCOL.json')
    hashes=read(PRIVATE/f'{stem}_hashes.json')
    path=PRIVATE/f'{stem}_probabilities.npy'; lp=PRIVATE/f'{stem}_logits.npy'
    assert sha(path)==hashes['probabilities_sha256'] and sha(lp)==hashes['logits_sha256']
    p=np.load(path,allow_pickle=False).astype(np.float64); logits=np.load(lp,allow_pickle=False)
    assert p.shape==logits.shape==(state['total'],2,8)
    assert np.isfinite(p).all() and np.isfinite(logits).all() and np.all((p>=0)&(p<=1))
    assert np.allclose(p[:,:,:7],softmax(logits[:,:,:7],axis=2),atol=2e-7,rtol=1e-6)
    assert np.allclose(p[:,:,7],expit(logits[:,:,7]),atol=2e-7,rtol=1e-6)
    return p

def compute_metrics():
    curves=OUT/'curve_coordinates'; curves.mkdir(exist_ok=True)
    y=np.load(PRIVATE/'labels.npy',allow_pickle=False)
    protocol=read(OUT/'P4_FROZEN_PROTOCOL.json')
    assert sha(OUT/'DEVELOPMENT_THRESHOLDS.csv')==protocol['thresholds']['sha256']
    thresholds=pd.read_csv(OUT/'DEVELOPMENT_THRESHOLDS.csv',float_precision='round_trip')
    assert len(thresholds)==60
    metrics=[]; counts=[]; bins=[]; diagnostics=[]; transported=[]
    for ident in read(OUT/'FIXED_MODEL_IDENTITIES.json'):
        if stopped(): return False
        family,seed,run=ident['model'],ident['training_seed'],ident['run']
        base={'model':family,'training_seed':seed,'run':run,'cohort':'Derm7pt'}
        p=complete_prob(ident)
        for mode,head,c,column in UNITS:
            score=p[:,c,column]; tags={**base,'input':mode,'head':head}
            metric,rel=binary(y,score)
            metrics.append({**tags,**metric}); bins.extend([{**tags,**b} for b in rel])
            for target in [.9,.95]:
                selected=thresholds[(thresholds.run==run)&(thresholds.training_seed==seed)&
                    (thresholds.model==family)&(thresholds['head']==head)&(thresholds.target_sensitivity==target)]
                assert len(selected)==1
                t=float(selected.iloc[0]['threshold']); assert np.isfinite(t)
                counts.append({**tags,'target_sensitivity':target,'threshold':t,**confusion_extended(y,score,t)})
                transported.append({**tags,'target_sensitivity':target,'original_threshold':t,'applied_threshold':t,
                    'source':'P3 saved own HAM validation threshold','reselected':False})
            fpr,tpr,rr=roc_curve(y,score,drop_intermediate=False)
            roc=pd.DataFrame({'point':np.arange(len(rr)),'fpr':fpr,'tpr':tpr,'threshold':rr})
            for key,value in tags.items(): roc[key]=value
            roc.to_csv(curves/f'{run}_{mode}_{head}_ROC.csv',index=False)
            precision,recall,rr=precision_recall_curve(y,score,drop_intermediate=False)
            pr=pd.DataFrame({'point':np.arange(len(recall)),'recall':recall,'precision':precision,'threshold':np.r_[rr,np.nan]})
            for key,value in tags.items(): pr[key]=value
            pr.to_csv(curves/f'{run}_{mode}_{head}_PR.csv',index=False)
        state=read(PRIVATE/f'{run}_Derm7pt_status.json')
        for mode,diag in state['diagnostics'].items():
            for key,row in diag.items():
                mean=row['sum']/row['n']
                diagnostics.append({**base,'input':mode,'diagnostic':key,'n':row['n'],'mean':mean,
                    'within_images_SD':np.sqrt(max(0,row['sum_sq']/row['n']-mean*mean)),
                    'note':'aggregated model readout, not independent localization accuracy'})
        log(f'METRICS_AND_CURVES_COMPLETE {run}')
    for name,rows in {'PER_RUN_METRICS':metrics,'THRESHOLD_CONFUSION_COUNTS':counts,
        'THRESHOLD_TRANSPORT_VERIFICATION':transported,'RELIABILITY_BINS':bins,'REGION_READOUT_DIAGNOSTICS':diagnostics}.items():
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    dump(OUT/'POINT_METRICS_STATUS.json',{'status':'COMPLETE','models':15,'binary_rows':len(metrics),
        'threshold_rows':len(counts),'curve_files':len(list(curves.glob('*.csv'))),
        'new_thresholds_selected':0,'new_calibrators_fitted':0})
    return True

def summarize():
    items=[('PER_RUN_METRICS',['cohort','input','head'],['roc_auc','average_precision','brier','binary_nll','ece15']),
        ('THRESHOLD_CONFUSION_COUNTS',['cohort','input','head','target_sensitivity'],['TP','FN','FP','TN','sensitivity','specificity','PPV','NPV'])]
    summary=[]; paired=[]
    for table,tags,metrics in items:
        df=pd.read_csv(OUT/f'{table}.csv',float_precision='round_trip')
        for key,group in df.groupby(tags,sort=True,dropna=False):
            tags_now=dict(zip(tags,key))
            for metric in metrics:
                matrix=group.pivot(index='training_seed',columns='model',values=metric).loc[SEEDS,MODELS]
                for model in MODELS:
                    a=matrix[model].to_numpy()
                    summary.append({'source':table,**tags_now,'model_or_contrast':model,'metric':metric,
                        'mean':np.mean(a),'sample_sd':np.std(a,ddof=1),'n_seeds':3,'defined_seeds':int(np.isfinite(a).sum())})
                for comparator in COMPARATORS:
                    a=(matrix['R_CLIN']-matrix[comparator]).to_numpy(); contrast=f'R_CLIN_minus_{comparator}'
                    summary.append({'source':table,**tags_now,'model_or_contrast':contrast,'metric':metric,
                        'mean':np.mean(a),'sample_sd':np.std(a,ddof=1),'n_seeds':3,'defined_seeds':int(np.isfinite(a).sum())})
                    for seed,value in zip(SEEDS,a):
                        paired.append({'source':table,**tags_now,'training_seed':seed,'contrast':contrast,'metric':metric,'difference':value})
    pd.DataFrame(summary).to_csv(OUT/'THREE_SEED_SUMMARY.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'PER_SEED_PAIRED_DIFFERENCES.csv',index=False)

def bootstrap():
    y=np.load(PRIVATE/'labels.npy',allow_pickle=False); cluster=np.load(PRIVATE/'clusters.npy',allow_pickle=False)
    binding=read(OUT/'COHORT_BINDING.json'); groups=binding['clusters']
    assert len(y)==len(cluster)==binding['included_rows'] and set(cluster)==set(range(groups))
    identities=read(OUT/'FIXED_MODEL_IDENTITIES.json'); calculators=[]; sources=[]
    for seed in SEEDS:
        fit=[]
        for family in MODELS:
            ident=next(i for i in identities if i['training_seed']==seed and i['model']==family)
            p=complete_prob(ident)
            fit.append([GroupedMetric(y,p[:,c,col]) for mode,head,c,col in UNITS])
            sources.append({'run':ident['run'],'sha256':sha(PRIVATE/f"{ident['run']}_Derm7pt_probabilities.npy")})
        calculators.append(fit)
    def evaluate(weights):
        return np.array([[[calc(weights) for calc in unit] for unit in fit] for fit in calculators])
    point=evaluate(np.ones(len(y))); table=pd.read_csv(OUT/'PER_RUN_METRICS.csv',float_precision='round_trip')
    for i,seed in enumerate(SEEDS):
        for j,family in enumerate(MODELS):
            for u,(mode,head,_,_) in enumerate(UNITS):
                row=table[(table.training_seed==seed)&(table.model==family)&(table.input==mode)&(table['head']==head)].iloc[0]
                assert np.allclose(point[i,j,u],row[['roc_auc','average_precision']].to_numpy(dtype=float),atol=1e-12,rtol=0)
    rng=np.random.Generator(np.random.PCG64(20261017)); accepted=rejected=0
    checkpoint=OUT/'BOOTSTRAP_PROGRESS.json'; draws_path=PRIVATE/'bootstrap_fixed_fits.npy'; shape=(10000,3,5,4,2)
    draws=np.lib.format.open_memmap(draws_path,mode='r+' if draws_path.exists() else 'w+',dtype='float64',shape=shape)
    assert draws.shape==shape and draws.dtype==np.float64
    if checkpoint.exists():
        state=read(checkpoint)
        assert state['sources']==sources and state['rng_seed']==20261017
        assert state['cluster_sha256']==sha(PRIVATE/'clusters.npy')
        accepted,rejected=state['accepted'],state['rejected']; rng.bit_generator.state=state['rng_state']
    dump(OUT/'BOOTSTRAP_INPUT_IDENTITY.json',{'sources':sources,'cluster_order':binding['cluster_order'],
        'cluster_sha256':sha(PRIVATE/'clusters.npy'),'label_sha256':sha(PRIVATE/'labels.npy'),
        'clusters':groups,'images':len(y),'verified_patient_level':False,
        'shared_draw_for_every_fit_condition_head':True,'fixed_seed_order':SEEDS,'model_order':MODELS})
    begin=time.time(); resumed=accepted
    def checkpoint_save():
        draws.flush()
        dump(checkpoint,{'accepted':accepted,'rejected':rejected,'target':10000,'rng_state':rng.bit_generator.state,
            'rng_seed':20261017,'sources':sources,'cluster_sha256':sha(PRIVATE/'clusters.npy')})
    while accepted<10000:
        if stopped(): break
        multiplicity=np.bincount(rng.integers(0,groups,groups),minlength=groups)
        weights=multiplicity[cluster].astype(float)
        if np.sum(weights*y)<=0 or np.sum(weights*(1-y))<=0:
            rejected+=1; continue
        result=evaluate(weights); assert np.isfinite(result).all()
        draws[accepted]=result; accepted+=1
        if accepted%100==0:
            checkpoint_save()
            update_progress(status='STATISTICS_RUNNING',statistics_status='RUNNING',bootstrap_accepted=accepted)
            if accepted%1000==0:
                log(f'CASE_CLUSTER_BOOTSTRAP {accepted}/10000 elapsed={time.time()-begin:.1f}s rejected={rejected}')
    checkpoint_save()
    if accepted<10000:
        dump(OUT/'STATISTICS_STATUS.json',{'status':'PARTIAL','accepted':accepted,'rejected':rejected})
        update_progress(status='PARTIAL',statistics_status='PARTIAL',bootstrap_accepted=accepted); return False
    rows=[]
    for comparator in COMPARATORS:
        a,b=MODELS.index('R_CLIN'),MODELS.index(comparator)
        delta=(draws[:,:,a]-draws[:,:,b]).mean(axis=1); point_delta=point[:,a]-point[:,b]
        for u,(mode,head,_,_) in enumerate(UNITS):
            for m,metric in enumerate(['roc_auc','average_precision']):
                primary=comparator in ['M0','XATTN'] and mode=='E_AVAILABLE' and head=='softmax_MEL' and metric=='average_precision'
                for coverage in ([.95,.975] if primary else [.95]):
                    alpha=(1-coverage)/2; low,high=np.quantile(delta[:,u,m],[alpha,1-alpha])
                    rows.append({'contrast':f'R_CLIN_minus_{comparator}','cohort':'Derm7pt','input':mode,'head':head,'metric':metric,
                        'role':'PRIMARY' if primary else 'SUPPORTING','mean_difference':point_delta[:,u,m].mean(),
                        'training_seed_sample_sd':point_delta[:,u,m].std(ddof=1),'coverage':coverage,'ci_low':low,'ci_high':high,
                        'draws':10000,'clusters':groups,'training_seeds':'42;2024;3407',
                        'CI_scope':'case/same-image cluster uncertainty conditional on three fixed fits; NOT verified patient-level',
                        'multiplicity_note':'approximate Bonferroni family95% for2primarycomparisons' if coverage==.975 else 'individual percentile interval'})
    pd.DataFrame(rows).to_csv(OUT/'CASE_CLUSTER_PAIRED_INTERVALS.csv',index=False)
    dump(OUT/'STATISTICS_STATUS.json',{'status':'COMPLETE','accepted':accepted,'rejected':rejected,
        'bootstrap_seconds_this_invocation':time.time()-begin,'resumed_from':resumed,
        'point_metrics_match_sklearn_tolerance':1e-12,'two_primary_AP_comparisons_have_95_and_97_5_intervals':True,
        'p_values_computed':False,'patient_level':False})
    update_progress(status='FINAL_VERIFICATION_PENDING',statistics_status='COMPLETE',bootstrap_accepted=10000)
    return True

def main():
    tests()
    assert read(OUT/'EVALUATION_STATUS.json')['status']=='COMPLETE'
    for entry in read(OUT/'P4_FROZEN_PROTOCOL.json')['manifest']:
        assert sha(entry['path'])==entry['sha256'],entry['path']
    state=OUT/'POINT_METRICS_STATUS.json'
    if not state.exists() or read(state)['status']!='COMPLETE':
        if not compute_metrics():
            update_progress(status='PARTIAL',statistics_status='PARTIAL'); return
    summarize(); bootstrap()

if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
