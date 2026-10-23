"""Descriptive HAM validation mechanism results; no new resampling or thresholds."""
import numpy as np
import pandas as pd
from scipy.special import softmax,expit
from sklearn.metrics import f1_score
from p5_common import *
from p5_metric_core import binary

def main():
    state=read(OUT/'TASK_B_STATUS.json');assert state['status'] in ['INFERENCE_COMPLETE','COMPLETE']
    y7=np.load(PRIVATE/'validation_labels.npy',allow_pickle=False);y=(y7==4).astype(int);assert len(y)==1513
    rows=[];diags=[];changes=[];reference_checks=[];permutation_reference=None
    for identity in read(OUT/'FIXED_MODEL_IDENTITIES.json'):
        run=identity['run'];tags={'model':identity['model'],'training_seed':identity['training_seed'],'run':run,'cohort':'HAM_val'}
        hashes=read(OUT/f'{run}_PREDICTION_FINGERPRINTS.json')
        for quantity in ['logits','probabilities']:assert sha(PRIVATE/f'{run}_{quantity}.npy')==hashes[f'{quantity}_sha256']
        p=np.load(PRIVATE/f'{run}_probabilities.npy',allow_pickle=False).astype(float)
        logits=np.load(PRIVATE/f'{run}_logits.npy',allow_pickle=False)
        assert p.shape==logits.shape==(1513,3,8) and np.isfinite(p).all() and np.isfinite(logits).all()
        assert np.allclose(p[:,:,:7],softmax(logits[:,:,:7],axis=-1),atol=2e-7,rtol=1e-6)
        assert np.allclose(p[:,:,7],expit(logits[:,:,7]),atol=2e-7,rtol=1e-6)
        inference=read(OUT/f'{run}_INFERENCE_SUMMARY.json')
        assert inference['status']=='COMPLETE' and inference['state_before']==inference['state_after']
        if permutation_reference is None:permutation_reference=inference['permutation_batch_fingerprints']
        else:assert permutation_reference==inference['permutation_batch_fingerprints'],'Permutation changed across models/seeds'
        # Existing P3 predictions are read only for native reproduction; never re-inferred.
        old_path=P3/f'private_predictions/{run}_HAM_val_probabilities.npy'
        old_logit=P3/f'private_predictions/{run}_HAM_val_logits.npy'
        old_hash=read(P3/f'private_predictions/{run}_HAM_val_hashes.json')
        assert sha(old_path)==old_hash['probabilities_sha256'] and sha(old_logit)==old_hash['logits_sha256']
        old=np.load(old_path,mmap_mode='r',allow_pickle=False)[:,0]
        old_l=np.load(old_logit,mmap_mode='r',allow_pickle=False)[:,0]
        assert np.allclose(p[:,0],old,atol=1e-5,rtol=1e-4)
        assert np.allclose(logits[:,0],old_l,atol=1e-5,rtol=1e-4)
        reference_checks.append({**tags,'probability_max_abs_error':float(np.max(np.abs(p[:,0]-old))),
            'logit_max_abs_error':float(np.max(np.abs(logits[:,0]-old_l))),
            'source':str(old_path),'source_sha256':old_hash['probabilities_sha256'],'passed':True})
        for ci,condition in enumerate(CONDITIONS):
            macro=float(f1_score(y7,p[:,ci,:7].argmax(1),labels=list(range(7)),average='macro'))
            for head,col in [('softmax_MEL',4),('endpoint_MEL',7)]:
                metric,_=binary(y,p[:,ci,col])
                rows.append({**tags,'condition':condition,'clinical_input':'NATIVE','head':head,
                    **metric,'seven_class_macro_f1':macro,'evidence_type':'validation spatial-dependence diagnostic'})
            for key,value in inference['diagnostics'][condition].items():
                assert value['n']==1513
                mean=value['sum']/value['n']
                diags.append({**tags,'condition':condition,'diagnostic':key,'n':value['n'],'mean':mean,
                    'within_images_population_sd':np.sqrt(max(0,value['sum_sq']/value['n']-mean**2)),
                    'min':value['min'],'max':value['max']})
            for head,cols in [('softmax_MEL',[4]),('endpoint_MEL',[7]),('seven_class',list(range(7)))]:
                a=p[:,ci][:,cols];b=p[:,0][:,cols];la=logits[:,ci][:,cols];lb=logits[:,0][:,cols]
                differences=a-b;logitdiff=la-lb
                changes.append({**tags,'condition':condition,'head':head,'reference':'NATIVE',
                    'probability_mean_signed_change':float(differences.mean()),
                    'probability_mean_absolute_change':float(np.abs(differences).mean()),
                    'probability_RMSE':float(np.sqrt(np.mean(differences**2))),
                    'probability_max_absolute_change':float(np.abs(differences).max()),
                    'logit_mean_absolute_change':float(np.abs(logitdiff).mean()),
                    'logit_max_absolute_change':float(np.abs(logitdiff).max()),
                    'seven_class_argmax_change_fraction':float(np.mean(p[:,ci,:7].argmax(1)!=p[:,0,:7].argmax(1)))})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'VALIDATION_SPATIAL_INTERVENTION.csv',index=False)
    pd.DataFrame(diags).to_csv(OUT/'REGION_SPATIAL_DIAGNOSTICS.csv',index=False)
    pd.DataFrame(changes).to_csv(OUT/'MODEL_OUTPUT_CHANGES.csv',index=False)
    pd.DataFrame(reference_checks).to_csv(OUT/'NATIVE_P3_REPRODUCTION.csv',index=False)
    metrics=['average_precision','roc_auc','brier','binary_nll','ece15','seven_class_macro_f1']
    summary=[];paired=[]
    for (model,head),group in frame.groupby(['model','head']):
        for metric in metrics:
            matrix=group.pivot(index='training_seed',columns='condition',values=metric).loc[SEEDS,CONDITIONS]
            direction='higher_is_better' if metric in ['average_precision','roc_auc','seven_class_macro_f1'] else 'lower_is_better'
            for condition in CONDITIONS:
                values=matrix[condition].to_numpy()
                summary.append({'model':model,'head':head,'metric':metric,'condition_or_contrast':condition,
                    'mean':values.mean(),'sample_sd':values.std(ddof=1),'n_seeds':3,'metric_improvement_direction':direction,
                    'interpretation':'descriptive; no confirmatory significance test'})
            for condition in CONDITIONS[1:]:
                values=(matrix.NATIVE-matrix[condition]).to_numpy()
                contrast=f'NATIVE_minus_{condition}'
                summary.append({'model':model,'head':head,'metric':metric,'condition_or_contrast':contrast,
                    'mean':values.mean(),'sample_sd':values.std(ddof=1),'n_seeds':3,'metric_improvement_direction':direction,
                    'interpretation':'positive favors NATIVE only for higher-is-better metrics; no significance test'})
                for s,v in zip(SEEDS,values):paired.append({'model':model,'head':head,'metric':metric,'contrast':contrast,'training_seed':s,'difference':v})
    pd.DataFrame(summary).to_csv(OUT/'SPATIAL_INTERVENTION_SUMMARY.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'SPATIAL_INTERVENTION_PER_SEED_DIFFERENCES.csv',index=False)
    diagnostic_summary=[]
    for source,tags in [('REGION_SPATIAL_DIAGNOSTICS',['model','condition','diagnostic']),('MODEL_OUTPUT_CHANGES',['model','condition','head'])]:
        f=pd.read_csv(OUT/f'{source}.csv',float_precision='round_trip')
        columns=['mean'] if source.startswith('REGION') else [c for c in f if c.startswith(('probability_','logit_','seven_class_'))]
        for key,group in f.groupby(tags):
            assert set(group.training_seed)==set(SEEDS)
            for metric in columns:
                a=group.set_index('training_seed').loc[SEEDS,metric].to_numpy()
                diagnostic_summary.append({'source':source,**dict(zip(tags,key)),'statistic':metric,'three_seed_mean':a.mean(),'sample_sd':a.std(ddof=1)})
    pd.DataFrame(diagnostic_summary).to_csv(OUT/'DIAGNOSTIC_THREE_SEED_SUMMARY.csv',index=False)
    state.update(status='COMPLETE',metric_rows=len(rows),new_bootstrap_draws=0,thresholds_selected=0,calibrators_fitted=0,
        all_six_native_reproduce_P3=True,same_permutations_all_six_models=True)
    dump(OUT/'TASK_B_STATUS.json',state);progress(task_B='COMPLETE',B_models_completed=6,B_current_run=None,B_images_completed=1513)
    log('TASK_B COMPLETE: six fixed models x 1513 HAM validation images x three conditions')

if __name__=='__main__':main()
