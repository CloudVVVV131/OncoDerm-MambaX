"""Additional contrasts from existing, read-only paired bootstrap metric arrays."""
import ast
import numpy as np
import pandas as pd
from p5_common import *

COMPARATORS=['M0','XATTN','B2S']

def literal_assignment(path,name):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    assert len(nodes)==1,(str(path),name)
    return ast.literal_eval(nodes[0].value),nodes[0].lineno

def stage(d,cohort,prefix,mode,interval_file,rng_seed):
    progress(task_A='RUNNING'); evidence=[]
    def record(p):
        evidence.append({'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size})
    for n in [f'{prefix}_common.py',f'{prefix}_statistics.py','FIXED_MODEL_IDENTITIES.json',
        'BOOTSTRAP_INPUT_IDENTITY.json','BOOTSTRAP_PROGRESS.json','STATISTICS_STATUS.json',
        'PER_RUN_METRICS.csv','THRESHOLD_CONFUSION_COUNTS.csv',interval_file]:record(d/n)
    seeds,sl=literal_assignment(d/f'{prefix}_common.py','SEEDS')
    models,ml=literal_assignment(d/f'{prefix}_common.py','MODELS')
    units,ul=literal_assignment(d/f'{prefix}_statistics.py','UNITS')
    shape,shl=literal_assignment(d/f'{prefix}_statistics.py','shape')
    assert seeds==SEEDS and models==MODELS and shape==(10000,3,5,4,2)
    expected=[(mode,'softmax_MEL',0,4),(mode,'endpoint_MEL',0,7),('E_NULL','softmax_MEL',1,4),('E_NULL','endpoint_MEL',1,7)]
    assert units==expected
    src=Path(d/f'{prefix}_statistics.py').read_text(encoding='utf-8-sig')
    assert "['roc_auc','average_precision']" in src.replace(' ','')
    assert 'for seed in SEEDS:' in src and 'for family in MODELS:' in src
    assert 'for fit in calculators' in src and 'for unit in fit' in src and 'for calc in unit' in src
    inp=read(d/'BOOTSTRAP_INPUT_IDENTITY.json'); state=read(d/'BOOTSTRAP_PROGRESS.json')
    assert state['accepted']==state['target']==10000 and state['rng_seed']==rng_seed
    assert read(d/'STATISTICS_STATUS.json')['status']=='COMPLETE'
    assert state['sources']==inp['sources'] and inp['shared_draw_for_every_fit_condition_head']
    identities=read(d/'FIXED_MODEL_IDENTITIES.json')
    ordered=[next(i['run'] for i in identities if i['training_seed']==s and i['model']==m) for s in SEEDS for m in MODELS]
    assert ordered==[r['run'] for r in inp['sources']]
    p=d/'private_predictions/bootstrap_fixed_fits.npy'; record(p)
    draws=np.load(p,mmap_mode='r',allow_pickle=False)
    assert draws.shape==shape and draws.dtype==np.float64 and not draws.flags.writeable
    assert np.isfinite(draws).all() and ((draws>=0)&(draws<=1)).all()
    table=pd.read_csv(d/'PER_RUN_METRICS.csv',float_precision='round_trip')
    table=table[table.cohort==cohort]
    assert len(table)==60 and not table.duplicated(['training_seed','model','input','head']).any()
    def values(model,unit,metric):
        u,h,_,_=units[unit]
        selected=table[(table.model==model)&(table.input==u)&(table['head']==h)].set_index('training_seed')
        return selected.loc[SEEDS,metric].to_numpy(dtype=float)
    # Reproduce ALL archived R_CLIN difference intervals as an independent axis/order check.
    old=pd.read_csv(d/interval_file,float_precision='round_trip'); checks=[]
    for _,row in old.iterrows():
        comparator=row.contrast.split('_minus_')[1]
        ui=next(i for i,x in enumerate(units) if x[:2]==(row.input,row['head']))
        mi=['roc_auc','average_precision'].index(row.metric)
        diff=(draws[:,:,models.index('R_CLIN'),ui,mi]-draws[:,:,models.index(comparator),ui,mi]).mean(1)
        alpha=(1-row.coverage)/2; q=np.quantile(diff,[alpha,1-alpha])
        point=(values('R_CLIN',ui,row.metric)-values(comparator,ui,row.metric)).mean()
        assert np.allclose(q,[row.ci_low,row.ci_high],atol=1e-12,rtol=0)
        assert abs(point-row.mean_difference)<1e-12
        checks.append({'contrast':row.contrast,'input':row.input,'head':row['head'],'metric':row.metric,
            'coverage':row.coverage,'max_interval_error':float(np.max(np.abs(q-[row.ci_low,row.ci_high])))})
    contrasts=[]; per_seed=[]
    for comparator in COMPARATORS:
        for ui,(condition,head,_,_) in enumerate(units):
            for mi,metric in enumerate(['roc_auc','average_precision']):
                delta=values('R_IMG',ui,metric)-values(comparator,ui,metric)
                boot=(draws[:,:,models.index('R_IMG'),ui,mi]-draws[:,:,models.index(comparator),ui,mi]).mean(1)
                focus=ui==0 and metric=='average_precision'
                tags={'cohort':cohort,'contrast':f'R_IMG_minus_{comparator}','input':condition,'head':head,'metric':metric}
                for s,x in zip(SEEDS,delta):per_seed.append({**tags,'training_seed':s,'difference':float(x)})
                for coverage in ([.95,1-.05/6] if focus else [.95]):
                    alpha=.05/(2*6) if coverage!=.95 else .025
                    low,high=np.quantile(boot,[alpha,1-alpha])
                    contrasts.append({**tags,'mean_difference':delta.mean(),'training_seed_sample_sd':delta.std(ddof=1),
                        'coverage':coverage,'quantile_low':alpha,'quantile_high':1-alpha,'ci_low':low,'ci_high':high,
                        'n_training_seeds':3,'valid_draws':10000,'clusters':inp['clusters'],
                        'sampling_unit':'patient' if cohort=='SIIM' else 'case/same-image cluster; not verified patient',
                        'role':'ADDITIONAL_FOCUS_AP' if focus else 'ADDITIONAL_SUPPORTING',
                        'scope':'conditional on fixed trained fits; post-result comparison selection',
                        'multiplicity':'approximate six-AP-comparison Bonferroni reference; does not undo selection' if coverage!=.95 else 'descriptive individual percentile'})
    reference=old.copy();reference['cohort']=cohort;reference['reference_source']=str(d/interval_file)
    reference['P5_action']='quoted unchanged; not replaced by R_IMG analyses'
    calibration=table[['model','training_seed','run','cohort','input','head','brier','binary_nll','ece15']].copy()
    calibration['improvement_direction']='lower Brier/NLL/ECE is better; original values only'
    th=pd.read_csv(d/'THRESHOLD_CONFUSION_COUNTS.csv',float_precision='round_trip');th=th[th.cohort==cohort].copy()
    th['P5_action']='original fixed thresholds and descriptive performance only; no fitting or selection'
    th['improvement_direction']='higher sensitivity/specificity/PPV/NPV is better at the SAME fixed policy; TP/TN higher and FN/FP lower'
    dump(OUT/f'{cohort}_BOOTSTRAP_REUSE_CHECK.json',{'status':'DIRECT_REUSE_VERIFIED','shape':list(draws.shape),
        'dtype':str(draws.dtype),'mmap_mode':'r','finite':True,'new_draws':0,'seed_order':seeds,'model_order':models,
        'units':units,'metric_order':['roc_auc','average_precision'],'accepted':state['accepted'],'rng_seed':rng_seed,
        'original_R_CLIN_intervals_reproduced':checks,'source_line_numbers':{'SEEDS':sl,'MODELS':ml,'UNITS':ul,'shape':shl},
        'original_bootstrap_input_identity':inp,'original_bootstrap_state':state,'manifest':evidence})
    log(f'TASK_A {cohort}: direct read-only reuse of 10000 draws verified; {len(contrasts)} interval rows')
    return contrasts,per_seed,reference,calibration,th

def main():
    outputs=[];errors=[]
    for d,cohort,prefix,mode,interval,rng in [(P3,'SIIM','p3','E_REAL','PATIENT_CLUSTER_PAIRED_INTERVALS.csv',20261016),
        (P4,'Derm7pt','p4','E_AVAILABLE','CASE_CLUSTER_PAIRED_INTERVALS.csv',20261017)]:
        if stopped():break
        try:outputs.append(stage(d,cohort,prefix,mode,interval,rng))
        except Exception as e:
            import traceback
            errors.append({'cohort':cohort,'error':str(e),'traceback':traceback.format_exc()})
    if outputs:
        pd.DataFrame([r for out in outputs for r in out[0]]).to_csv(OUT/'REGION_CONTRASTS.csv',index=False)
        pd.DataFrame([r for out in outputs for r in out[1]]).to_csv(OUT/'REGION_CONTRASTS_PER_SEED.csv',index=False)
        for i,name in [(2,'ORIGINAL_RCLIN_COMPARISONS'),(3,'ORIGINAL_CALIBRATION_METRICS'),(4,'ORIGINAL_FIXED_THRESHOLD_PERFORMANCE')]:
            pd.concat([o[i] for o in outputs],ignore_index=True).to_csv(OUT/f'{name}.csv',index=False)
    status='COMPLETE' if len(outputs)==2 else 'BLOCKED' if errors else 'PARTIAL'
    dump(OUT/'TASK_A_STATUS.json',{'status':status,'cohorts_completed':len(outputs),'errors':errors,
        'direct_reuse':len(outputs),'new_draws':0,'external_inference':0})
    progress(task_A=status,A_cohorts_completed=len(outputs))
    if errors:log('TASK_A input issue recorded; independent task B may continue')

if __name__=='__main__':main()
