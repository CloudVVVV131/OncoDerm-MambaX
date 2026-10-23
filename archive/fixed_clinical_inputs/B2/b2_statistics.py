"""B2 paired input statistics, 2026-10-12, PCG64 seed20261013.
CPU only. Reads new fixed-model predictions; never imports B1 or any model.
Thresholds: same-run/score HAM val >=90/95% sensitivity, best specificity.
Bootstrap: 10000 valid patient draws; fixed three runs, no model/calibration refit.
"""
from b2_common import *
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve,brier_score_loss,log_loss
from threadpoolctl import threadpool_limits
try:
    import statsmodels
    import statsmodels.api as sm
except ImportError:
    statsmodels=None
SCORES=['prob_mel','prob_mel_endpoint']
METRICS=['roc_auc','average_precision','brier','binary_nll','ece15']
TARGETS=[.9,.95]
SEEDS=[42,2024,3407]
EPS=1e-7


class GroupedMetric:
    """Pure tie-grouped AUC/AP implementation copied from tested B1, no B1 import."""
    def __init__(self,y,score):
        self.order=np.argsort(-score,kind='mergesort')
        s=score[self.order]
        self.starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
        self.y=y[self.order].astype(np.float64)
    def __call__(self,weights):
        w=weights[self.order]
        positive=np.add.reduceat(w*self.y,self.starts)
        negative=np.add.reduceat(w*(1-self.y),self.starts)
        p,n=positive.sum(),negative.sum()
        if p<=0 or n<=0:return np.array([np.nan,np.nan])
        tp=np.cumsum(positive);fp=np.cumsum(negative)
        precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
        return np.array([np.sum(negative*(tp-.5*positive))/(p*n),np.sum(positive*precision)/p])


def reference(y,p,w=None):
    return np.array([roc_auc_score(y,p,sample_weight=w),average_precision_score(y,p,sample_weight=w)])


def threshold(y,p,target):
    assert set(np.unique(y))=={0,1}
    order=np.argsort(-p,kind='stable'); s=p[order];yy=y[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp=np.cumsum(yy)[ends]; fp=(ends+1)-tp
    sensitivity=tp/y.sum()
    valid=np.flatnonzero(sensitivity>=target)
    assert len(valid)
    best_fp=np.min(fp[valid]);candidates=valid[fp[valid]==best_fp]
    index=candidates[np.argmax(s[ends[candidates]])]
    return float(s[ends[index]])


def confusion(y,p,t):
    positive=p>=t
    tp=int(np.sum(positive&(y==1)));fn=int(np.sum(~positive&(y==1)))
    fp=int(np.sum(positive&(y==0)));tn=int(np.sum(~positive&(y==0)))
    ratio=lambda a,b:float(a/b) if b else None
    return dict(TP=tp,FN=fn,FP=fp,TN=tn,sensitivity=ratio(tp,tp+fn),specificity=ratio(tn,tn+fp),
        PPV=ratio(tp,tp+fp),NPV=ratio(tn,tn+fn),prevalence=float(np.mean(y)))


def calibration(y,p):
    if statsmodels is None:return dict(calibration_status='unavailable_missing_statsmodels',calibration_intercept=None,calibration_slope=None)
    x=np.log(np.clip(p,EPS,1-EPS)/(1-np.clip(p,EPS,1-EPS)))
    if len(np.unique(y))<2 or np.ptp(x)==0:
        return dict(calibration_status='unavailable_degenerate_design',calibration_intercept=None,calibration_slope=None)
    if x[y==0].max()<=x[y==1].min() or x[y==1].max()<=x[y==0].min():
        return dict(calibration_status='unavailable_complete_or_quasi_separation',calibration_intercept=None,calibration_slope=None)
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            fit=sm.GLM(y,np.column_stack([np.ones(len(x)),x]),family=sm.families.Binomial()).fit(maxiter=200,tol=1e-10)
        assert fit.converged and np.isfinite(fit.params).all()
        assert not any('separation' in str(w.message).lower() for w in ws)
        return dict(calibration_status='converged_unpenalized_binomial_MLE_evaluation_only',
            calibration_intercept=float(fit.params[0]),calibration_slope=float(fit.params[1]))
    except Exception as exc:
        return dict(calibration_status='unavailable_'+type(exc).__name__,calibration_intercept=None,calibration_slope=None)


def point(y,p):
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    clipped=np.clip(p,EPS,1-EPS)
    brier=float(np.mean((p-y)**2))
    nll=float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped)))
    bins=np.minimum((p*15).astype(int),14)
    reliability=[];ece=0.
    for k in range(15):
        index=bins==k;n=int(index.sum())
        mean=float(p[index].mean()) if n else None;rate=float(y[index].mean()) if n else None
        if n:ece+=n/len(y)*abs(mean-rate)
        reliability.append(dict(bin_index=k,lower=k/15,upper=(k+1)/15,count=n,mean_probability=mean,observed_rate=rate))
    return dict(images=len(y),positive_images=int(y.sum()),prevalence=float(y.mean()),
        roc_auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),
        brier=brier,binary_nll=nll,ece15=ece,**calibration(y,p)),reliability


def tests():
    records=[];rng=np.random.Generator(np.random.PCG64(20261014))
    cases=[('ties',[0,1,0,1,0,1],[.1,.5,.5,.5,.9,.9],[1]*6),
           ('constant',[0,1,0,1],[.4]*4,[1,1,1,1]),
           ('nonuniform',[0,1,0,1,1],[.7,.9,.3,.7,.1],[3,1,5,2,4]),
           ('zero_weights',[0,1,0,1,0,1],[.9,.8,.7,.7,.3,.1],[0,0,3,2,0,1]),
           ('two_classes',[0,1],[.2,.8],[2,3])]
    for i in range(12):
        y=np.tile([0,1],10);p=rng.random(20) if i%2 else rng.integers(0,5,20)/4;w=rng.integers(0,5,20);w[:2]+=1
        cases.append((f'random{i}',y,p,w))
    patient=np.array([0,0,1,2,2,3,3,3]);multi=np.bincount([0,0,2,3],minlength=4)
    cases.append(('patient_repeat',[0,1,0,1,0,0,1,1],[.1,.8,.4,.8,.7,.2,.5,.5],multi[patient]))
    for name,y,p,w in cases:
        y=np.array(y);p=np.array(p);w=np.array(w,dtype=float)
        got=GroupedMetric(y,p)(w);ref=reference(y,p,w);ix=np.repeat(np.arange(len(y)),w.astype(int));explicit=reference(y[ix],p[ix])
        error=float(max(abs(got-ref).max(),abs(got-explicit).max()));assert error<=1e-10
        b1=float(np.average((p-y)**2,weights=w));b2=float(brier_score_loss(y,p,sample_weight=w))
        assert abs(b1-b2)<=1e-10
        records.append(dict(test=name,max_abs_error=error,passed=True))
        for target in TARGETS:
            th=threshold(y,p,target)
            valid=[(confusion(y,p,t)['specificity'],t) for t in np.unique(p) if confusion(y,p,t)['sensitivity']>=target]
            expected=max(valid)[1];assert th==expected
    for name,y,w in [('all_positive',np.ones(4),np.ones(4)),('all_negative',np.zeros(4),np.ones(4)),('effective_one_class',np.array([0,1,1,0]),np.array([0,1,3,0])),('zero',np.array([0,1,0,1]),np.zeros(4))]:
        assert np.isnan(GroupedMetric(y,np.arange(4)/4)(w)).all()
        records.append(dict(test=name,max_abs_error=None,passed=True))
    assert confusion(np.array([0,1]),np.array([.1,.2]),1)['PPV'] is None
    assert confusion(np.array([0,1]),np.array([.1,.2]),0)['NPV'] is None
    result=dict(time=now(),all_passed=True,cases=records,
        thresholds_match_brute_force=True,undefined_denominators_are_null=True,
        provenance='GroupedMetric arithmetic copied from B1 pure class; no import of B1 entrypoint')
    if (OUT/'STATISTICAL_TESTS.json').exists():
        assert js(OUT/'STATISTICAL_TESTS.json')['cases']==records
    else:save_json('STATISTICAL_TESTS.json',result)


def load_predictions():
    assert js(OUT/'INFERENCE_STATUS.json')['status']=='COMPLETE'
    alignment=pd.read_csv(LOCAL/'metadata/SIIM_alignment.csv',dtype={'image_name':str,'patient_id':str})
    patients=sorted(alignment.patient_id.unique());lookup={p:i for i,p in enumerate(patients)}
    cluster=alignment.patient_id.map(lookup).to_numpy(dtype=int)
    y=alignment.target.to_numpy(dtype=int)
    sources=[];units={}
    for item in js(OUT/'PREDICTION_MANIFEST.json'):
        path=Path(item['path']);assert sha(path)==item['sha256']
        frame=pd.read_csv(path,dtype={'image_id':str},float_precision='round_trip')
        assert set(frame.condition)==set(item['conditions'])
        for condition in item['conditions']:
            sub=frame[frame.condition==condition].copy()
            assert len(sub)==item['images'] and sub.image_id.is_unique
            if item['dataset']=='SIIM':
                assert sub.image_id.tolist()==alignment.image_name.tolist() and np.array_equal(sub.label.to_numpy(),y)
            for score in SCORES:
                p=sub[score].to_numpy(dtype=np.float64)
                assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
            units[(item['model'],int(item['training_seed']),item['dataset'],condition)]=sub
        sources.append(dict(path=str(path),sha256=item['sha256'],bytes=path.stat().st_size))
    assert len(units)==24
    return units,y,cluster,len(patients),sources


def aggregate(units):
    output=[];reliability=[];thresholds=[];confusions=[];point_lookup={};threshold_lookup={}
    for (model,seed,dataset,condition),frame in units.items():
        y=frame.label.to_numpy(dtype=int)
        for score in SCORES:
            p=frame[score].to_numpy(dtype=np.float64)
            base=dict(model=model,training_seed=seed,dataset=dataset,condition=condition,score=score)
            met,rel=point(y,p);record=dict(**base,**met);output.append(record)
            point_lookup[(model,seed,dataset,condition,score)]=record
            reliability.extend(dict(**base,**r) for r in rel)
            if dataset=='HAM_val':
                for target in TARGETS:
                    value=threshold(y,p,target);cm=confusion(y,p,value)
                    thresholds.append(dict(model=model,training_seed=seed,score=score,target_sensitivity=target,
                        threshold=value,source='HAM_val_NATIVE',selection='max_specificity_then_largest_threshold',**cm))
                    threshold_lookup[(model,seed,score,target)]=value
    for (model,seed,dataset,condition),frame in units.items():
        for score in SCORES:
            for target in TARGETS:
                value=threshold_lookup[(model,seed,score,target)]
                confusions.append(dict(model=model,training_seed=seed,dataset=dataset,condition=condition,score=score,
                    target_sensitivity=target,threshold=value,threshold_source='HAM_val_NATIVE',
                    **confusion(frame.label.to_numpy(dtype=int),frame[score].to_numpy(dtype=np.float64),value)))
    save_csv('B2_METRICS.csv',output);save_csv('RELIABILITY_BINS.csv',reliability)
    save_csv('DEVELOPMENT_THRESHOLDS.csv',thresholds);save_csv('THRESHOLD_CONFUSION_COUNTS.csv',confusions)
    summary=[]
    for model in ['B2','B6']:
        for condition in ['E_NULL','E_REAL']:
            for score in SCORES:
                for metric in METRICS:
                    x=np.array([point_lookup[(model,s,'SIIM',condition,score)][metric] for s in SEEDS])
                    summary.append(dict(model=model,condition=condition,score=score,metric=metric,training_seeds='42|2024|3407',
                        n_runs=3,mean=float(x.mean()),sample_sd_ddof1=float(x.std(ddof=1)),minimum=float(x.min()),maximum=float(x.max())))
    save_csv('THREE_RUN_CONDITION_SUMMARY.csv',summary)
    contrasts=[]
    for model in ['B2','B6']:
        for seed in SEEDS:
            for score in SCORES:
                n=point_lookup[(model,seed,'SIIM','E_NULL',score)];r=point_lookup[(model,seed,'SIIM','E_REAL',score)]
                for metric in METRICS:
                    contrasts.append(dict(model=model,training_seed=seed,score=score,metric=metric,E_NULL=n[metric],E_REAL=r[metric],
                        difference_E_REAL_minus_E_NULL=r[metric]-n[metric],CI_status='bootstrap_in_paired_table' if metric in METRICS[:3] else 'point_only_as_protocol'))
    save_csv('PAIRED_POINT_DIFFERENCES.csv',contrasts)
    return point_lookup,threshold_lookup


def curves(units):
    tests=[]
    with (OUT/'ROC_COORDINATES.csv').open('x',encoding='utf-8',newline='') as rf,(OUT/'PR_COORDINATES.csv').open('x',encoding='utf-8',newline='') as pf:
        rw=csv.DictWriter(rf,fieldnames=['model','training_seed','condition','score','point_index','false_positive_rate','true_positive_rate','threshold'])
        pw=csv.DictWriter(pf,fieldnames=['model','training_seed','condition','score','point_index','recall','precision','threshold'])
        rw.writeheader();pw.writeheader()
        for (model,seed,dataset,condition),frame in units.items():
            if dataset!='SIIM':continue
            y=frame.label.to_numpy(dtype=int)
            for score in SCORES:
                p=frame[score].to_numpy(dtype=float)
                fpr,tpr,th=roc_curve(y,p,drop_intermediate=False)
                precision,recall,pth=precision_recall_curve(y,p,drop_intermediate=False)
                assert (fpr[0],tpr[0])==(0,0) and (fpr[-1],tpr[-1])==(1,1)
                assert recall[0]==1 and recall[-1]==0 and precision[-1]==1
                assert abs(-np.sum(np.diff(recall)*precision[:-1])-average_precision_score(y,p))<=1e-10
                base=dict(model=model,training_seed=seed,condition=condition,score=score)
                for i in range(len(fpr)):rw.writerow(dict(**base,point_index=i,false_positive_rate=fpr[i],true_positive_rate=tpr[i],threshold=th[i]))
                for i in range(len(recall)):pw.writerow(dict(**base,point_index=i,recall=recall[i],precision=precision[i],threshold=pth[i] if i<len(pth) else ''))
                tests.append(dict(**base,roc_points=len(fpr),pr_points=len(recall),PR_x='recall',PR_y='precision',passed=True))
    save_json('CURVE_TESTS.json',tests)


def history_diagnostic(units):
    manifest=csv_rows(NAV/'INPUT_MANIFEST.csv');results=[];sources=[]
    for entry in entries():
        candidates=[r for r in manifest if r['role']=='STAB_repeated' and entry['run_id']+'_external_predictions.csv' in r['path']]
        assert len(candidates)==1
        item=candidates[0];actual=sha(item['path'])
        assert actual==item['sha256'], 'Historical diagnostic input hash changed: '+item['path']
        old=pd.read_csv(item['path'],dtype={'image_id':str},usecols=['image_id']+SCORES,float_precision='round_trip').set_index('image_id')
        new=units[(entry['model'],int(entry['training_seed']),'SIIM','E_NULL')]
        assert set(old.index)==set(new.image_id)
        old=old.loc[new.image_id]
        for score in SCORES:
            delta=new[score].to_numpy()-old[score].to_numpy()
            results.append(dict(model=entry['model'],training_seed=int(entry['training_seed']),score=score,
                mean_absolute_difference=float(np.abs(delta).mean()),maximum_absolute_difference=float(np.abs(delta).max()),
                mean_signed_new_FP32_minus_archived=float(delta.mean()),scope='diagnostic_only_not_B2_input_effect',
                historical_precision='historical settings differ; no assertion of bitwise equality'))
        sources.append(dict(path=item['path'],sha256=actual,bytes=Path(item['path']).stat().st_size))
    save_csv('HISTORICAL_NULL_DIAGNOSTIC.csv',results)
    save_csv('HISTORICAL_DIAGNOSTIC_SOURCES.csv',sources)


def bootstrap(units,y,cluster,npatients,lookup,thresh,resume=False):
    keys=[(model,seed,condition,score) for model in ['B2','B6'] for seed in SEEDS for condition in ['E_NULL','E_REAL'] for score in SCORES]
    ps=[units[(m,s,'SIIM',c)][h].to_numpy(dtype=np.float64) for m,s,c,h in keys]
    rank=[GroupedMetric(y,p) for p in ps]
    rng_test=np.random.Generator(np.random.PCG64(20261014));checks=[]
    for draw in range(3):
        mult=np.bincount(rng_test.integers(0,npatients,size=npatients),minlength=npatients);w=mult[cluster].astype(float)
        for key,p,calc in zip(keys,ps,rank):
            err=float(np.max(abs(calc(w)-reference(y,p,w))));assert err<=1e-10
            checks.append(dict(draw=draw,key=list(key),max_abs_error=err,passed=True))
    if not resume:save_json('REAL_WEIGHT_TESTS.json',dict(all_passed=True,count=len(checks),validation_seed=20261014,checks=checks))
    count_groups=np.bincount(cluster,minlength=npatients);pos_groups=np.bincount(cluster,weights=y,minlength=npatients)
    neg_groups=count_groups-pos_groups
    brier_groups=np.column_stack([np.bincount(cluster,weights=(p-y)**2,minlength=npatients) for p in ps])
    tp_groups=[];tn_groups=[]
    for key,p in zip(keys,ps):
        model,seed,condition,score=key
        for target in TARGETS:
            pred=p>=thresh[(model,seed,score,target)]
            tp_groups.append(np.bincount(cluster,weights=pred*y,minlength=npatients))
            tn_groups.append(np.bincount(cluster,weights=(~pred)*(1-y),minlength=npatients))
    tp_groups=np.column_stack(tp_groups);tn_groups=np.column_stack(tn_groups)
    grouped_checks=[]
    for draw in range(3):
        mult=np.bincount(rng_test.integers(0,npatients,size=npatients),minlength=npatients)
        w=mult[cluster].astype(float);pos=mult@pos_groups;neg=mult@neg_groups
        for j,(key,p) in enumerate(zip(keys,ps)):
            actual=float((mult@brier_groups[:,j])/(pos+neg))
            expected=float(np.average((p-y)**2,weights=w))
            error=abs(actual-expected)
            for policy,target in enumerate(TARGETS):
                pred=p>=thresh[(key[0],key[1],key[3],target)]
                error=max(error,abs((mult@tp_groups[:,j*2+policy])/pos-np.sum(w*pred*y)/pos),
                    abs((mult@tn_groups[:,j*2+policy])/neg-np.sum(w*(~pred)*(1-y))/neg))
            assert error<=1e-10
            grouped_checks.append(dict(draw=draw,key=list(key),max_abs_error=float(error),passed=True))
    if not resume:save_json('GROUPED_BRIER_THRESHOLD_TESTS.json',dict(all_passed=True,checks=grouped_checks))
    n=10000;values=np.empty((n,len(keys),3));rates=np.empty((n,len(keys),2,2))
    rng=np.random.Generator(np.random.PCG64(20261013));completed=attempts=rejected=0;prior=0.
    if resume:
        state_path=sorted((LOCAL/'bootstrap').glob('state_*.json'))[-1];state=js(state_path)
        assert state['keys']==[list(k) for k in keys]
        assert state['protocol_sha256']==sha(OUT/'B2_FROZEN_PROTOCOL.json')
        assert sha(state_path.with_suffix('.npz'))==state['array_sha256']
        with np.load(state_path.with_suffix('.npz'),allow_pickle=False) as saved:
            completed=state['completed'];values[:completed]=saved['values'];rates[:completed]=saved['rates']
        rng.bit_generator.state=state['rng_state'];attempts=state['attempts'];rejected=state['rejected'];prior=state['elapsed_seconds']
    start=time.perf_counter()
    inference_seconds=js(OUT/'INFERENCE_STATUS.json')['elapsed_seconds']
    while completed<n:
        attempts+=1
        mult=np.bincount(rng.integers(0,npatients,size=npatients),minlength=npatients)
        pos=mult@pos_groups;neg=mult@neg_groups
        if pos<=0 or neg<=0:rejected+=1;continue
        w=mult[cluster].astype(float)
        for j,calc in enumerate(rank):values[completed,j,:2]=calc(w)
        values[completed,:,2]=(mult@brier_groups)/(pos+neg)
        rates[completed,:,:,0]=(mult@tp_groups).reshape(len(keys),2)/pos
        rates[completed,:,:,1]=(mult@tn_groups).reshape(len(keys),2)/neg
        assert np.isfinite(values[completed]).all() and np.isfinite(rates[completed]).all()
        completed+=1
        if completed==100 or completed%1000==0:
            elapsed=prior+time.perf_counter()-start
            folder=LOCAL/'bootstrap';folder.mkdir(exist_ok=True)
            stem=f'state_{completed:05}'
            with (folder/(stem+'.npz')).open('xb') as f:np.savez_compressed(f,values=values[:completed],rates=rates[:completed])
            state=dict(completed=completed,attempts=attempts,rejected=rejected,elapsed_seconds=elapsed,keys=keys,
                rng_state=rng.bit_generator.state,protocol_sha256=sha(OUT/'B2_FROZEN_PROTOCOL.json'),
                array_sha256=sha(folder/(stem+'.npz')))
            save_json('local_only/bootstrap/'+stem+'.json',state)
            if completed==100:
                estimate=elapsed/completed*n
                save_json('STATISTICS_PILOT_100.json',dict(time=now(),elapsed_seconds=elapsed,projected_bootstrap_seconds=estimate,
                    first100_included=True,seed=20261013,same_stream=True))
                if inference_seconds+estimate>14400:
                    save_json('STATISTICS_STATUS.json',dict(status='PARTIAL_TIME_GATE',completed=completed,estimated_bootstrap_seconds=estimate));return
            log(f'B2 bootstrap {completed}/10000; rejected={rejected}; elapsed={elapsed:.1f}s')
            if inference_seconds+elapsed>14400 and completed<n:
                save_json('STATISTICS_STATUS.json',dict(status='PARTIAL_TIME_GATE',completed=completed));return
    key_index={key:i for i,key in enumerate(keys)}
    pair=[];mean=[];threshold_ci=[];threshold_mean=[];threshold_absolute=[]
    for model in ['B2','B6']:
        for score in SCORES:
            for j,metric in enumerate(METRICS[:3]):
                deltas=[];points=[]
                for seed in SEEDS:
                    ni=key_index[(model,seed,'E_NULL',score)];ri=key_index[(model,seed,'E_REAL',score)]
                    d=values[:,ri,j]-values[:,ni,j];deltas.append(d)
                    npoint=lookup[(model,seed,'SIIM','E_NULL',score)][metric];rpoint=lookup[(model,seed,'SIIM','E_REAL',score)][metric];points.append(rpoint-npoint)
                    lo,hi=np.quantile(d,[.025,.975],method='linear')
                    pair.append(dict(model=model,training_seed=seed,score=score,metric=metric,E_NULL=npoint,E_REAL=rpoint,
                        difference=rpoint-npoint,CI95_lower=float(lo),CI95_upper=float(hi),valid_replicates=n,estimand='E_REAL_minus_E_NULL_fixed_run',role='supporting',p_value='not_computed'))
                mean_delta=np.mean(deltas,axis=0);lo,hi=np.quantile(mean_delta,[.025,.975],method='linear')
                primary=score=='prob_mel' and metric=='average_precision'
                family=np.quantile(mean_delta,[.0125,.9875],method='linear') if primary else [None,None]
                mean.append(dict(model=model,score=score,metric=metric,training_seeds='42|2024|3407',
                    mean_difference=float(np.mean(points)),sample_SD_of_seed_differences=float(np.std(points,ddof=1)),
                    CI95_lower=float(lo),CI95_upper=float(hi),primary_AP_CI975_lower=float(family[0]) if primary else None,
                    primary_AP_CI975_upper=float(family[1]) if primary else None,valid_replicates=n,
                    estimand='conditional_on_fixed_three_runs_E_REAL_minus_E_NULL',role='primary_AP' if primary else 'supporting',p_value='not_computed'))
            for policy,target in enumerate(TARGETS):
                for j,metric in enumerate(['sensitivity','specificity']):
                    deltas=[];points=[]
                    for seed in SEEDS:
                        ni=key_index[(model,seed,'E_NULL',score)];ri=key_index[(model,seed,'E_REAL',score)]
                        d=rates[:,ri,policy,j]-rates[:,ni,policy,j];deltas.append(d)
                        val=thresh[(model,seed,score,target)]
                        pn=confusion(y,ps[ni],val)[metric];pr=confusion(y,ps[ri],val)[metric];points.append(pr-pn)
                        lo,hi=np.quantile(d,[.025,.975],method='linear')
                        threshold_ci.append(dict(model=model,training_seed=seed,score=score,target_sensitivity=target,threshold=val,metric=metric,
                            E_NULL=pn,E_REAL=pr,difference=pr-pn,CI95_lower=float(lo),CI95_upper=float(hi),valid_replicates=n,
                            conditioning='fixed_run_and_selected_HAM_validation_threshold'))
                        for condition,index,value in [('E_NULL',ni,pn),('E_REAL',ri,pr)]:
                            lo2,hi2=np.quantile(rates[:,index,policy,j],[.025,.975],method='linear')
                            threshold_absolute.append(dict(model=model,training_seed=seed,score=score,target_sensitivity=target,condition=condition,metric=metric,
                                point=value,CI95_lower=float(lo2),CI95_upper=float(hi2),valid_replicates=n,conditioning='selected_validation_threshold_fixed'))
                    lo,hi=np.quantile(np.mean(deltas,axis=0),[.025,.975],method='linear')
                    threshold_mean.append(dict(model=model,score=score,target_sensitivity=target,metric=metric,training_seeds='42|2024|3407',
                        mean_difference=float(np.mean(points)),CI95_lower=float(lo),CI95_upper=float(hi),valid_replicates=n,
                        conditioning='fixed_three_runs_and_selected_validation_thresholds'))
    save_csv('PAIRED_PATIENT_CLUSTER_CI.csv',pair);save_csv('FIXED_THREE_RUN_MEAN_CI.csv',mean)
    save_csv('THRESHOLD_PAIRED_CI.csv',threshold_ci);save_csv('THRESHOLD_FIXED_THREE_RUN_CI.csv',threshold_mean)
    save_csv('THRESHOLD_ABSOLUTE_CI.csv',threshold_absolute)
    save_json('STATISTICS_STATUS.json',dict(status='COMPLETE',completed_valid=n,attempts=attempts,rejected_single_class=rejected,
        bootstrap_elapsed_seconds=prior+time.perf_counter()-start,score_vectors=len(keys),main_pairwise_CI_rows=len(pair),
        fixed_mean_CI_rows=len(mean),threshold_pairwise_rows=len(threshold_ci),threshold_mean_rows=len(threshold_mean),
        generator='PCG64',seed=20261013,calibration_bootstrap_refits=0,threshold_bootstrap_reselection=0,p_values=False))


def main(resume=False):
    with threadpool_limits(limits=1):
        guard_sources()
        if not resume:
            save_json('STATISTICAL_ENVIRONMENT.json',dict(time=now(),executable=sys.executable,python=sys.version,
                numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                statsmodels=statsmodels.__version__ if statsmodels else None,workers=1,BLAS_threads=1,torch_imported=False))
            tests()
        units,y,cluster,npatients,sources=load_predictions()
        if not resume:
            save_csv('NEW_PREDICTION_SOURCE_MANIFEST.csv',sources)
            lookup,thresh=aggregate(units)
            history_diagnostic(units)
            curves(units)
            log('B2 points, validation thresholds, historical NULL diagnostics and full coordinates complete')
        else:
            lookup={}
            for r in csv_rows(OUT/'B2_METRICS.csv'):
                for m in METRICS:r[m]=float(r[m])
                lookup[(r['model'],int(r['training_seed']),r['dataset'],r['condition'],r['score'])]=r
            thresh={(r['model'],int(r['training_seed']),r['score'],float(r['target_sensitivity'])):float(r['threshold']) for r in csv_rows(OUT/'DEVELOPMENT_THRESHOLDS.csv')}
        bootstrap(units,y,cluster,npatients,lookup,thresh,resume)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--tests-only',action='store_true');args=parser.parse_args()
    if args.tests_only:tests()
    else:main(args.resume)
