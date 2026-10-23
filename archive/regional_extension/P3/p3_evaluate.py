"""P3 unified fixed-best inference with chunk checkpoints; no training imports."""
import gc
import time
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from p3_common import *
from p3_eval_data import EvaluationData, NULL, worker_init
from p3_eval_model import FixedCandidate, OncoDermMambaX, cache_image, cached_forward


def loader(ds, start=0, end=None):
    return DataLoader(ds, sampler=list(range(start, len(ds) if end is None else end)),
        batch_size=8, num_workers=2, pin_memory=True, prefetch_factor=1,
        persistent_workers=False, worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(ds.training_seed*1000+17))


def outputs(out):
    return (torch.cat([out['logits'],out['mel_logits'][:,None]],1),
            torch.cat([out['logits'].softmax(1),out['mel_logits'].sigmoid()[:,None]],1))


def merge_diag(total, out, n):
    for key, value in out.get('readout_diagnostics',{}).items():
        value = value.detach().float()
        if value.ndim == 0:
            value = value.expand(n)
        if value.ndim == 1:
            value = value[:,None]
        value = value.cpu().numpy().astype(float)
        for j in range(value.shape[1]):
            k = f'{key}_{j}'
            v = value[:,j]
            row = total.setdefault(k, {'n':0,'sum':0.,'sum_sq':0.})
            row['n'] += len(v); row['sum'] += float(v.sum()); row['sum_sq'] += float((v*v).sum())


@torch.inference_mode()
def verify_cache(model, ds, run):
    batch = next(iter(loader(ds,0,8)))
    x, meta = batch[0].cuda(), batch[1].cuda()
    cached = cache_image(model,x)
    rows=[]
    for mode,m in [('NATIVE_real_fields',meta),('E_NULL',NULL.cuda().expand(len(x),-1))]:
        full, fast = outputs(model(x,m)), outputs(cached_forward(model,cached,m))
        for quantity, a,b in zip(['logits','probabilities'],full,fast):
            rows.append({'condition':mode,'quantity':quantity,
                'max_abs_error':float((a-b).abs().max()),'passed':torch.allclose(a,b,atol=1e-5,rtol=1e-4),
                'both_heads_included':True})
    passed=all(r['passed'] for r in rows)
    dump(OUT / f'{run}_CACHE_CHECK.json', {'status':'PASSED' if passed else 'FULL_FORWARD_FALLBACK',
        'fixed_validation_indices':list(range(8)), 'atol':1e-5,'rtol':1e-4,'checks':rows,
        'recomputes_clinical_readout_and_CMGF_per_condition':True})
    return passed


@torch.inference_mode()
def infer(model, ds, identity, use_cache):
    run=identity['run']; cohort=ds.cohort
    stem=f'{run}_{cohort}'
    state_path=PRIVATE / f'{stem}_status.json'
    lp=PRIVATE / f'{stem}_logits.npy'; pp=PRIVATE / f'{stem}_probabilities.npy'
    modes=['E_REAL','E_NULL'] if cohort=='SIIM' else ['NATIVE']
    start=0; diagnostics={mode:{} for mode in modes}; prior_seconds=0.
    if state_path.exists():
        state=read(state_path)
        assert state['best_sha256']==identity['sha256'] and state['modes']==modes
        if state['status']=='COMPLETE':
            assert state['next_index']==len(ds) and lp.exists() and pp.exists()
            hash_path=PRIVATE / f'{stem}_hashes.json'
            observed={'logits_sha256':sha(lp),'probabilities_sha256':sha(pp)}
            if hash_path.exists():
                assert read(hash_path)==observed
            else:
                dump(hash_path,observed)
            dump(OUT / f'{stem}_INFERENCE_SUMMARY.json',state)
            return True
        start=state['next_index']; diagnostics=state['diagnostics']; prior_seconds=state['elapsed_seconds']
    shape=(len(ds),len(modes),8)
    logits=np.lib.format.open_memmap(lp,mode='r+' if lp.exists() else 'w+',dtype='float32',shape=shape)
    prob=np.lib.format.open_memmap(pp,mode='r+' if pp.exists() else 'w+',dtype='float32',shape=shape)
    assert logits.shape==prob.shape==shape
    begin=time.time()
    for images,metadata,labels,index in loader(ds,start):
        x=images.cuda(non_blocking=True); cached=cache_image(model,x) if use_cache else None
        for c,mode in enumerate(modes):
            meta=NULL.cuda().expand(len(x),-1) if mode=='E_NULL' else metadata.cuda(non_blocking=True)
            out=cached_forward(model,cached,meta) if use_cache else model(x,meta)
            values,scores=outputs(out)
            assert torch.isfinite(values).all() and torch.isfinite(scores).all()
            logits[index.numpy(),c]=values.cpu().numpy(); prob[index.numpy(),c]=scores.cpu().numpy()
            merge_diag(diagnostics[mode],out,len(x))
        last=int(index[-1])+1
        if last%800==0 or last==len(ds) or stopped():
            logits.flush(); prob.flush()
            dump(state_path,{'status':'COMPLETE' if last==len(ds) else 'PARTIAL',
                'next_index':last,'total':len(ds),'modes':modes,'cohort':cohort,'run':run,
                'best_sha256':identity['sha256'],'elapsed_seconds':prior_seconds+time.time()-begin,
                'diagnostics':diagnostics,'cache':use_cache})
            log(f'INFERENCE {stem} {last}/{len(ds)} elapsed={time.time()-begin:.1f}s')
            update_progress(status='EVALUATING',evaluation_status='RUNNING',current=f'{stem}:{last}/{len(ds)}')
        if stopped() and last<len(ds):
            return False
    del logits,prob
    dump(PRIVATE / f'{stem}_hashes.json',{'logits_sha256':sha(lp),'probabilities_sha256':sha(pp)})
    dump(OUT / f'{stem}_INFERENCE_SUMMARY.json',read(state_path))
    return True


def validation_check(ds, identity, selection):
    p=np.load(PRIVATE / f"{identity['run']}_HAM_val_probabilities.npy",allow_pickle=False)[:,0].astype(float)
    y=ds.labels
    got={'macro_f1':float(f1_score(y,p[:,:7].argmax(1),average='macro',labels=list(range(7)))),
         'mel_auc':float(roc_auc_score(y==4,p[:,4])), 'mel_ap':float(average_precision_score(y==4,p[:,4]))}
    checks={k:{'observed':got[k],'original':selection[k], 'absolute_difference':abs(got[k]-selection[k])}
            for k in got if k in selection}
    passed=all(r['absolute_difference']<=1e-7 for r in checks.values())
    dump(OUT / f"{identity['run']}_VALIDATION_REPRODUCTION.json",{'status':'PASSED' if passed else 'MISMATCH',
        'checks':checks,'tolerance':1e-7,'best_reselected':False,'no_adjustments':True})
    return passed


def main():
    protocol=read(OUT / 'P3_FROZEN_PROTOCOL.json')
    for entry in protocol['manifest']:
        assert sha(entry['path'])==entry['sha256'], entry['path']
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    completed=[]; discrepancies=[]
    for identity in protocol['models']:
        if stopped():
            update_progress(status='PARTIAL',evaluation_status='PARTIAL',evaluation_completed=len(completed))
            return
        assert sha(identity['path'])==identity['sha256']
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            ckpt=torch.load(identity['path'],map_location='cpu',weights_only=True)
        assert ckpt['epoch']==identity['best_epoch']
        cfg=ckpt['config']['original_config']
        cfg['model']['pretrained']=False
        cfg['training_seed']=identity['training_seed']
        model=(FixedCandidate(cfg,identity['model'],identity['training_seed'])
               if identity['model'] in ['R_IMG','R_CLIN','XATTN'] else OncoDermMambaX(cfg))
        model.load_state_dict(ckpt['model'],strict=True)
        selection=ckpt['selection_validation']
        model.requires_grad_(False).eval().cuda()
        assert all(not m.training for m in model.modules())
        assert all(p.dtype==torch.float32 for p in model.parameters())
        dump(OUT / f"{identity['run']}_EVALUATION_MODEL.json",{**identity,'strict_load':True,
            'source_model':str(SOURCE/'model_code/lesionmamba/models/oncoderm_mambax.py'),
            'candidate_source':str(OUT/'p3_eval_model.py') if isinstance(model,FixedCandidate) else None,
            'precision':'FP32','TF32':False,'eval_mode':True})
        del ckpt; gc.collect()
        ds=EvaluationData(cfg,'HAM_val')
        use_cache=verify_cache(model,ds,identity['run'])
        for cohort in ['HAM_val','HAM_test','SIIM']:
            if cohort!='HAM_val':
                ds=EvaluationData(cfg,cohort)
            if not infer(model,ds,identity,use_cache):
                update_progress(status='PARTIAL',evaluation_status='PARTIAL',evaluation_completed=len(completed))
                return
            if cohort=='HAM_val' and not validation_check(ds,identity,selection):
                discrepancies.append(identity['run'])
        completed.append(identity['run'])
        dump(OUT/'EVALUATION_STATUS.json',{'status':'RUNNING','completed':completed,
             'validation_discrepancies':discrepancies})
        del model,ds; gc.collect(); torch.cuda.empty_cache()
    dump(OUT/'EVALUATION_STATUS.json',{'status':'COMPLETE','completed':completed,'models':15,
        'validation_discrepancies':discrepancies,'all_predictions_generated_in_P3':True})
    update_progress(status='STATISTICS_PENDING',evaluation_status='COMPLETE',evaluation_completed=15)


if __name__=='__main__':
    raise SystemExit('Use p3_eval_entry.py to isolate spawned workers.')
