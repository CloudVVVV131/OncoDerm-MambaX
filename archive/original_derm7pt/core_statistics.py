"""Prespecified additional six-original-model analysis; never invokes a model."""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from core_common import *
from core_metrics import GroupedMetric, binary

METRICS=['average_precision','roc_auc','brier','binary_nll','ece15']

def aggregate(frame,keys,columns):
    rows=[]
    for key,g in frame.groupby(keys,sort=False):
        if not isinstance(key,tuple):key=(key,)
        assert set(g.training_seed.astype(int))==set(SEEDS) and len(g)==3
        for c in columns:
            values=g[c].to_numpy(dtype=float)
            valid=np.isfinite(values)
            rows.append(dict(**dict(zip(keys,key)),metric=c,n_runs=3,n_defined=int(valid.sum()),
                mean=float(values.mean()) if valid.all() else None,
                sample_sd=float(values.std(ddof=1)) if valid.all() else None,
                status='DEFINED' if valid.all() else 'UNDEFINED_DENOMINATOR'))
    return rows

def points(protocol,y,scores):
    per=[];reliability=[];counts=[];roc=[];pr=[]
    thresholds=table(OUT/'THRESHOLD_IDENTITY_JOIN.csv') if (OUT/'THRESHOLD_IDENTITY_JOIN.csv').exists() else []
    for fi,family in enumerate(FAMILIES):
        for si,seed in enumerate(SEEDS):
            ident=next(m for m in protocol['models'] if m['model']==family and m['training_seed']==seed)
            for ci,cond in enumerate(CONDITIONS):
                for hi,head in enumerate(HEADS):
                    p=scores[fi,si,ci,:,hi]
                    key=dict(model=family,training_seed=seed,run_id=ident['run_id'],condition=cond,score=head)
                    value,bins=binary(y,p)
                    per.append(dict(**key,**value,prevalence=float(y.mean()),undefined_reason=''))
                    reliability.extend(dict(**key,**b) for b in bins)
                    fpr,tpr,rt=roc_curve(y,p,drop_intermediate=False)
                    precision,recall,pt=precision_recall_curve(y,p,drop_intermediate=False)
                    roc.extend(dict(**key,coordinate=i,false_positive_rate=float(a),true_positive_rate=float(b),threshold=float(t)) for i,(a,b,t) in enumerate(zip(fpr,tpr,rt)))
                    pr.extend(dict(**key,coordinate=i,recall=float(a),precision=float(b),threshold=float(pt[i]) if i<len(pt) else None) for i,(a,b) in enumerate(zip(recall,precision)))
                    for row in thresholds:
                        if row['run_id']!=ident['run_id'] or row['score']!=head:continue
                        assert row['checkpoint_sha256']==ident['sha256'] and int(row['training_seed'])==seed
                        t=float(row['threshold']);pred=p>=t
                        tp=int(np.sum(pred&(y==1)));fn=int(np.sum(~pred&(y==1)))
                        fp=int(np.sum(pred&(y==0)));tn=int(np.sum(~pred&(y==0)))
                        counts.append(dict(**key,target_sensitivity=float(row['target_sensitivity']),threshold=t,
                            threshold_original_text=row['threshold'],TP=tp,FN=fn,FP=fp,TN=tn,
                            sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),
                            PPV=tp/(tp+fp) if tp+fp else None,NPV=tn/(tn+fn) if tn+fn else None,
                            undefined_reason=';'.join(s for s in ['PPV:no_predicted_positive' if not tp+fp else '', 'NPV:no_predicted_negative' if not tn+fn else ''] if s),
                            prevalence=float(y.mean()),threshold_source='B2 HAM_val_NATIVE; no target-domain threshold selection'))
    csv_save(OUT/'PER_RUN_METRICS.csv',per)
    csv_save(OUT/'THREE_RUN_SUMMARY.csv',aggregate(pd.DataFrame(per),['model','condition','score'],METRICS))
    csv_save(OUT/'RELIABILITY_BINS.csv',reliability)
    csv_save(OUT/'ROC_COORDINATES.csv',roc);csv_save(OUT/'PR_COORDINATES.csv',pr)
    if thresholds:
        assert len(counts)==48
        csv_save(OUT/'THRESHOLD_COUNTS.csv',counts)
        csv_save(OUT/'THRESHOLD_SUMMARY.csv',aggregate(pd.DataFrame(counts),['model','condition','score','target_sensitivity'],['TP','FN','FP','TN','sensitivity','specificity','PPV','NPV']))
        progress(thresholds='COMPLETE',threshold_source_rows=24,threshold_application_rows=48)
    else:progress(thresholds='MISSING_SOURCE_OR_IDENTITY')
    # Keep separate cohorts and age-coverage policies; never subtract their APs.
    siim=pd.DataFrame(table(OUT/'SIIM_B2_AGGREGATES_SOURCE.csv'))
    for c in METRICS:siim[c]=siim[c].astype(float)
    siim['training_seed']=siim.training_seed.astype(int)
    siim['model']='ORIG_'+siim.model
    cross=[]
    for cohort,frame,age,n,positive in [('SIIM',siim,'E_REAL includes observed age; E_NULL has no observed clinical fields',33126,584),
                                     ('Derm7pt',pd.DataFrame(per),'E_AVAILABLE includes sex/location only; age missing in all cases',len(y),int(y.sum()))]:
        for r in aggregate(frame,['model','condition','score'],METRICS):
            cross.append(dict(dataset=cohort,images=n,positive_images=positive,prevalence=positive/n,age_coverage=age,precision='FP32_TF32_off',**r))
    csv_save(OUT/'ORIGINAL_CORE_CROSS_COHORT_SUMMARY.csv',cross)
    dump(OUT/'POINT_OUTPUTS.json',dict(status='COMPLETE',per_run_rows=len(per),ROC_vectors=24,PR_vectors=24,ROC_rows=len(roc),PR_rows=len(pr),reliability_rows=len(reliability),threshold_rows=len(counts),time=now()))
    return per

def bootstrap(protocol,y,scores):
    cluster=np.load(PRIVATE/'clusters.npy',allow_pickle=False)
    k=protocol['cohort']['clusters'];assert sorted(set(cluster))==list(range(k))
    engines=[[[[GroupedMetric(y,scores[f,s,c,:,h]) for h in range(2)] for c in range(2)] for s in range(3)] for f in range(2)]
    point=np.empty((2,3,2,2,2),dtype=np.float64)
    max_error=0.
    for f in range(2):
        for s in range(3):
            for c in range(2):
                for h in range(2):
                    p=scores[f,s,c,:,h];got=engines[f][s][c][h](np.ones(len(y)))
                    ref=np.array([roc_auc_score(y,p),average_precision_score(y,p)])
                    max_error=max(max_error,float(np.max(abs(got-ref))))
                    assert np.allclose(got,ref,atol=1e-12,rtol=0)
                    point[f,s,c,h]=got
    # Validate tie handling and repeated-cluster weights without fitting anything.
    test_y=np.array([0,1,0,1,0,1]);test_p=np.array([.2,.8,.2,.7,.7,.7]);w=np.array([2,1,0,3,1,2])
    got=GroupedMetric(test_y,test_p)(w)
    ref=np.array([roc_auc_score(test_y,test_p,sample_weight=w),average_precision_score(test_y,test_p,sample_weight=w)])
    assert np.allclose(got,ref,atol=1e-12,rtol=0)
    dump(OUT/'METRIC_TESTS.json',dict(status='PASS',real_vector_count=24,max_unweighted_error=max_error,ties_and_weighted_test_max_error=float(np.max(abs(got-ref))),AP='noninterpolated average precision; NOT trapezoidal PR area'))
    rng=np.random.Generator(np.random.PCG64(20261021))
    checkpoint=PRIVATE/'bootstrap_checkpoint.json';path=PRIVATE/'bootstrap_fixed_fits.npy'
    identities={m['tag']:read(OUT/(m['tag']+'_INFERENCE.json'))['prediction_sha256'] for m in protocol['models']}
    signature=dict(predictions=identities,protocol_sha256=sha(OUT/'ORIGINALCORE_PROTOCOL.json'),labels_sha256=sha(PRIVATE/'labels.npy'),clusters_sha256=sha(PRIVATE/'clusters.npy'))
    if checkpoint.exists():
        state=read(checkpoint);assert state['input_identity']==signature
        valid=state['valid'];attempted=state['attempted'];rng.bit_generator.state=state['rng_state']
        fits=np.lib.format.open_memmap(path,mode='r+')
    else:
        valid=attempted=0
        fits=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(10000,2,3,2,2,2))
    start=time.perf_counter()
    while valid<10000:
        if time.time()>=DEADLINE-120:break
        multiplicities=np.bincount(rng.integers(0,k,size=k),minlength=k)
        weights=multiplicities[cluster]
        attempted+=1
        if np.dot(weights,y)==0 or np.dot(weights,1-y)==0:continue
        for f in range(2):
            for s in range(3):
                for c in range(2):
                    for h in range(2):fits[valid,f,s,c,h]=engines[f][s][c][h](weights)
        valid+=1
        if valid%100==0:
            fits.flush()
            dump(checkpoint,dict(valid=valid,attempted=attempted,rejected_single_class=attempted-valid,rng_state=rng.bit_generator.state,input_identity=signature,updated_at=now()))
            progress(statistics='RUNNING',valid_bootstrap=valid)
        if valid%1000==0:log(f'Bootstrap {valid}/10000; elapsed={time.perf_counter()-start:.1f}s')
    fits.flush()
    dump(checkpoint,dict(valid=valid,attempted=attempted,rejected_single_class=attempted-valid,rng_state=rng.bit_generator.state,input_identity=signature,updated_at=now()))
    dump(OUT/'BOOTSTRAP_STATUS.json',dict(status='COMPLETE' if valid==10000 else 'PARTIAL',valid=valid,attempted=attempted,rejected_single_class=attempted-valid,seconds=time.perf_counter()-start,
       RNG='PCG64',seed=20261021,shape=list(fits.shape),axes=['replicate','family','training_seed','condition','head','metric'],family_order=FAMILIES,training_seed_order=SEEDS,condition_order=CONDITIONS,head_order=HEADS,metric_order=['roc_auc','average_precision'],
       cluster_count=k,unit=protocol['cohort']['resampling_unit'],patient_level=False,quantile_method='linear',fixed_training_runs=True,seed_resampling=False,array_sha256=sha(path)))
    if valid!=10000:
        progress(status='PARTIAL',statistics='PARTIAL',valid_bootstrap=valid);return
    assert np.isfinite(fits).all()
    def ci(values,level):
        a=(1-level)/2
        lo,hi=np.quantile(values,[a,1-a],method='linear')
        return dict(confidence=level,lower=float(lo),upper=float(hi),replicates=10000,unit='case/same-image cluster',conditional_on='three fixed training runs',method='percentile; linear quantile')
    effects=[];absolute=[];config=[];seed_differences=[]
    for f,family in enumerate(FAMILIES):
        for h,head in enumerate(HEADS):
            for mi,metric in enumerate(['roc_auc','average_precision']):
                per_seed=point[f,:,0,h,mi]-point[f,:,1,h,mi]
                samples=(fits[:,f,:,0,h,mi]-fits[:,f,:,1,h,mi]).mean(axis=1)
                levels=[.95,.975] if h==0 and mi==1 else [.95]
                for level in levels:
                    effects.append(dict(model=family,score=head,metric=metric,contrast='E_AVAILABLE - E_NULL',mean=float(per_seed.mean()),seed_difference_sample_sd=float(per_seed.std(ddof=1)),
                        analysis_role='two focus additional input AP comparisons' if h==0 and mi==1 else 'supporting',**ci(samples,level)))
                for si,seed in enumerate(SEEDS):seed_differences.append(dict(contrast=family+':E_AVAILABLE-E_NULL',training_seed=seed,condition='paired_inputs',score=head,metric=metric,difference=float(per_seed[si])))
                if h==0:
                    absolute.append(dict(model=family,condition='E_AVAILABLE',score=head,metric=metric,mean=float(point[f,:,0,h,mi].mean()),sample_sd=float(point[f,:,0,h,mi].std(ddof=1)),**ci(fits[:,f,:,0,h,mi].mean(axis=1),.95)))
    for c,cond in enumerate(CONDITIONS):
        for h,head in enumerate(HEADS):
            for mi,metric in enumerate(['roc_auc','average_precision']):
                per_seed=point[1,:,c,h,mi]-point[0,:,c,h,mi]
                samples=(fits[:,1,:,c,h,mi]-fits[:,0,:,c,h,mi]).mean(axis=1)
                config.append(dict(contrast='ORIG_B6 - ORIG_B2',condition=cond,score=head,metric=metric,mean=float(per_seed.mean()),seed_difference_sample_sd=float(per_seed.std(ddof=1)),analysis_role='supporting complete-configuration comparison; NOT single-factor ablation',**ci(samples,.95)))
                for si,seed in enumerate(SEEDS):seed_differences.append(dict(contrast='ORIG_B6-ORIG_B2',training_seed=seed,condition=cond,score=head,metric=metric,difference=float(per_seed[si])))
    csv_save(OUT/'INPUT_EFFECTS.csv',effects);csv_save(OUT/'FIXED_MODEL_INTERVALS.csv',absolute)
    csv_save(OUT/'ORIGINAL_B6_VS_B2.csv',config);csv_save(OUT/'PER_SEED_DIFFERENCES.csv',seed_differences)
    progress(status='STATISTICS_COMPLETE',statistics='COMPLETE',valid_bootstrap=valid)

def main():
    protocol=read(OUT/'ORIGINALCORE_PROTOCOL.json')
    assert read(OUT/'INFERENCE_SUMMARY.json')['status']=='COMPLETE'
    assert time.time()<DEADLINE
    progress(statistics='RUNNING')
    y=np.load(PRIVATE/'labels.npy',allow_pickle=False)
    scores=np.empty((2,3,2,len(y),2),dtype=np.float64)
    for f,family in enumerate(FAMILIES):
        for s,seed in enumerate(SEEDS):
            tag=f'{family}_s{seed}';meta=read(OUT/(tag+'_INFERENCE.json'))
            assert meta['status']=='COMPLETE' and meta['rows']==len(y)
            assert sha(PRIVATE/(tag+'.npy'))==meta['prediction_sha256']
            scores[f,s]=np.load(PRIVATE/(tag+'.npy'),allow_pickle=False)
    assert np.isfinite(scores).all() and scores.min()>=0 and scores.max()<=1
    points(protocol,y,scores)
    bootstrap(protocol,y,scores)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        progress(status='BLOCKED',statistics='BLOCKED',error=repr(exc));raise
