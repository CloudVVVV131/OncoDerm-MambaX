"""Record P5 scope and bounded existing-source identities before new analysis."""
import shutil
from p5_common import *

def main():
    assert not (OUT/'P5_FROZEN_PROTOCOL.json').exists()
    free=shutil.disk_usage(OUT).free; assert free>=2*1024**3
    p3=read(P3/'P3_FROZEN_PROTOCOL.json'); p4=read(P4/'P4_FROZEN_PROTOCOL.json')
    all_models=read(P3/'FIXED_MODEL_IDENTITIES.json')
    assert all_models==p3['models']==p4['models']==read(P4/'FIXED_MODEL_IDENTITIES.json')
    identities=[r for r in all_models if r['model'] in ['R_IMG','R_CLIN']]
    assert len(identities)==6
    val=next(x for x in p3['manifest'] if x['path'].endswith('ham10000_seed42_val.csv'))
    assert sha(val['path'])==val['sha256']
    observed=[]
    def record(path,expected=None):
        path=Path(path); digest=sha(path)
        if expected: assert digest==expected,str(path)
        observed.append({'path':str(path),'sha256':digest,'bytes':path.stat().st_size})
    record(OUT/'TASK_PROMPT.txt')
    for d,stage,report in [(P3,'P3','P3_FINAL_EVAL_REPORT.md'),(P4,'P4','P4_EXTERNAL_REPORT.md')]:
        for name in [f'{stage}_FROZEN_PROTOCOL.json',f'{stage}_PROGRESS.json',report,'FIXED_MODEL_IDENTITIES.json',
                     'BOOTSTRAP_INPUT_IDENTITY.json','BOOTSTRAP_PROGRESS.json','STATISTICS_STATUS.json']:
            record(d/name)
        assert read(d/f'{stage}_PROGRESS.json')['status']=='COMPLETE'
    for identity in identities:
        record(identity['path'],identity['sha256'])
        record(identity['config_snapshot'],identity['effective_config_sha256'])
    for item in p3['manifest']:
        path=Path(item['path'])
        if path.is_relative_to(P3/'source'):
            record(path,item['sha256']); record(SOURCE/path.relative_to(P3/'source'),item['sha256'])
    record(val['path'],val['sha256'])
    record(P3/'HAM_val_INPUT_IDENTITY.json')
    record(P3/'private_predictions/HAM_val_alignment.csv',read(P3/'HAM_val_INPUT_IDENTITY.json')['alignment_sha256'])
    record(P3/'p3_eval_data.py'); record(P3/'p3_common.py')
    protocol={'frozen_at':now(),'started_at':START.isoformat(),'deadline':datetime.fromtimestamp(DEADLINE,ZONE).isoformat(),
        'status':'FROZEN','scope':'Additional analyses and HAM validation mechanism diagnostics, not rewritten P3/P4 primary analyses',
        'new_training_runs':0,'task_A':{'source_roots':[str(P3),str(P4)],'preferred_source':'read-only mmap of original bootstrap_fixed_fits.npy',
            'contrasts':['R_IMG_minus_M0','R_IMG_minus_XATTN','R_IMG_minus_B2S'],'seed_order':SEEDS,'model_order':MODELS,
            'shape':[10000,3,5,4,2],'metric_order':['roc_auc','average_precision'],
            'point_source':'PER_RUN_METRICS.csv, per-run metric differences first, then three-seed mean and sample SD ddof1',
            'intervals':'95% percentile on saved paired difference draws; six focus AP comparisons additionally 1-0.05/6 coverage',
            'six_comparison_quantiles':[0.05/(2*6),1-0.05/(2*6)],
            'qualification':'additional, post-result comparison selection; correction does not remove selection bias; not original confirmatory endpoints',
            'preserve_original_R_CLIN_comparisons':True,'no_external_inference':True},
        'task_B':{'models':identities,'cohort':'HAM_val','images':1513,'input':'NATIVE','conditions':CONDITIONS,
            'ham_validation_csv':val,'image_root':str(ROOT/'project/revision/data/HAM10000/images'),
            'path_source':'P3 p3_eval_data.py EvaluationData HAM branch: REV/data/HAM10000/images; P3 frozen split file',
            'intervention_site':'region_tokens: after original resize/sigmoid/3x3 replicate-padded soft morphology; before feature-weighted pooling',
            'SHIFT_HALF':'all three completed region maps torch.roll(H//2,W//2) in spatial axes',
            'PERMUTE_FIXED':'uid=int.from_bytes(SHA256(exact_image_id UTF8)[:8],little,unsigned); PCG64(SeedSequence([20261018,uid])).permutation(H*W); all regions gather same index',
            'same_rule_all_models_and_seeds':True,'region_order':['interior','boundary','context'],
            'mass_rtol':1e-5,'native_check':{'first_images':8,'atol':1e-5,'rtol':1e-4},
            'alpha':'recomputed from changed tokens, not frozen','state':'all parameters and buffers unchanged',
            'caching':'batch-local backbone/fused/mask/original_pool/clinical_embedding once; no disk feature cache',
            'preprocessing':p3['preprocessing'],'precision':p3['precision'],'batch':8,'workers':2,
            'metrics':p3['metrics'],'bootstrap':False,'threshold_selection':False,'calibration_fitting':False,
            'interpretation':'validation fixed-model spatial-dependence diagnostic, not localization accuracy, causal lesion explanation or new external validation'},
        'prohibited':['training','reselection','new architecture','new seeds','external forwarding','cloud','system changes','manuscript edits'],
        'disk_free_bytes_at_start':free,'reserved_output_budget_bytes':2*1024**3,'manifest':observed}
    dump(OUT/'P5_FROZEN_PROTOCOL.json',protocol); dump(OUT/'FIXED_MODEL_IDENTITIES.json',identities)
    PRIVATE.mkdir(exist_ok=True)
    progress(status='RUNNING',task_A='PENDING',task_B='PENDING',B_models_completed=0)
    log('P5 protocol frozen; only six HAM-validation fixed models are authorized for new inference')

if __name__=='__main__': main()
