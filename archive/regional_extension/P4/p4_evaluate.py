"""Strict fixed-best Derm7pt inference only; no training or previous-cohort driver."""
import copy
import gc
import random
import time
import sys
import platform
import numpy as np
import torch
from torch.utils.data import DataLoader
from p4_common import *
from p4_data import EvaluationData, NULL, worker_init
from p4_eval_model import FixedCandidate, OncoDermMambaX, cache_image, cached_forward

def loader(ds,start=0,end=None):
    return DataLoader(ds,sampler=list(range(start,len(ds) if end is None else end)),
        batch_size=8,num_workers=2,pin_memory=True,prefetch_factor=1,persistent_workers=False,
        worker_init_fn=worker_init,generator=torch.Generator().manual_seed(ds.training_seed*1000+17))

def outputs(out):
    return (torch.cat([out['logits'],out['mel_logits'][:,None]],1),
            torch.cat([out['logits'].softmax(1),out['mel_logits'].sigmoid()[:,None]],1))

def merge_diag(total,out,n):
    for key,value in out.get('readout_diagnostics',{}).items():
        value=value.detach().float()
        if value.ndim==0: value=value.expand(n)
        if value.ndim==1: value=value[:,None]
        value=value.cpu().numpy().astype(float)
        for j in range(value.shape[1]):
            row=total.setdefault(f'{key}_{j}',{'n':0,'sum':0.,'sum_sq':0.})
            v=value[:,j]; row['n']+=len(v); row['sum']+=float(v.sum()); row['sum_sq']+=float((v*v).sum())

@torch.inference_mode()
def verify_cache(model,ds,run):
    images,metadata,_=next(iter(loader(ds,0,min(8,len(ds)))))
    x=images.cuda(); cached=cache_image(model,x); rows=[]
    for mode,m in [('E_AVAILABLE',metadata.cuda()),('E_NULL',NULL.cuda().expand(len(x),-1))]:
        full,fast=outputs(model(x,m)),outputs(cached_forward(model,cached,m))
        for name,a,b in zip(['logits','probabilities'],full,fast):
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            rows.append({'condition':mode,'quantity':name,'max_abs_error':float((a-b).abs().max()),
                'passed':bool(torch.allclose(a,b,atol=1e-5,rtol=1e-4)),'both_heads_included':True})
    passed=all(r['passed'] for r in rows)
    dump(OUT/f'{run}_CACHE_CHECK.json',{'status':'PASSED' if passed else 'FULL_FORWARD_FALLBACK',
        'fixed_external_indices':list(range(min(8,len(ds)))),'atol':1e-5,'rtol':1e-4,'checks':rows,
        'recomputes_clinical_readout_and_CMGF_per_condition':True})
    return passed

@torch.inference_mode()
def infer(model,ds,identity,use_cache):
    run=identity['run']; stem=f'{run}_Derm7pt'; modes=['E_AVAILABLE','E_NULL']
    state_path=PRIVATE/f'{stem}_status.json'; lp=PRIVATE/f'{stem}_logits.npy'; pp=PRIVATE/f'{stem}_probabilities.npy'
    start=0; diagnostics={m:{} for m in modes}; prior_seconds=0.
    protocol_hash=sha(OUT/'P4_FROZEN_PROTOCOL.json')
    if state_path.exists():
        state=read(state_path)
        assert state['best_sha256']==identity['sha256'] and state['protocol_sha256']==protocol_hash
        assert state['modes']==modes and state['total']==len(ds)
        if state['status']=='COMPLETE':
            assert state['next_index']==len(ds)
            assert read(PRIVATE/f'{stem}_hashes.json')=={'logits_sha256':sha(lp),'probabilities_sha256':sha(pp)}
            return True
        start=state['next_index']; diagnostics=state['diagnostics']; prior_seconds=state['elapsed_seconds']
    else:
        assert not lp.exists() and not pp.exists(),'Uncheckpointed output; inspect rather than overwrite'
    shape=(len(ds),2,8)
    logits=np.lib.format.open_memmap(lp,mode='r+' if lp.exists() else 'w+',dtype='float32',shape=shape)
    prob=np.lib.format.open_memmap(pp,mode='r+' if pp.exists() else 'w+',dtype='float32',shape=shape)
    assert logits.shape==prob.shape==shape
    begin=time.time()
    for images,metadata,index in loader(ds,start):
        x=images.cuda(non_blocking=True); cached=cache_image(model,x) if use_cache else None
        for c,mode in enumerate(modes):
            meta=NULL.cuda().expand(len(x),-1) if mode=='E_NULL' else metadata.cuda(non_blocking=True)
            out=cached_forward(model,cached,meta) if use_cache else model(x,meta)
            values,scores=outputs(out)
            assert torch.isfinite(values).all() and torch.isfinite(scores).all()
            logits[index.numpy(),c]=values.cpu().numpy(); prob[index.numpy(),c]=scores.cpu().numpy()
            merge_diag(diagnostics[mode],out,len(x))
        last=int(index[-1])+1
        if last%128==0 or last==len(ds) or stopped():
            logits.flush(); prob.flush()
            dump(state_path,{'status':'COMPLETE' if last==len(ds) else 'PARTIAL','next_index':last,'total':len(ds),
                'modes':modes,'cohort':'Derm7pt','run':run,'best_sha256':identity['sha256'],
                'protocol_sha256':protocol_hash,'elapsed_seconds':prior_seconds+time.time()-begin,
                'diagnostics':diagnostics,'cache':use_cache})
            log(f'INFERENCE {run} {last}/{len(ds)} elapsed={time.time()-begin:.1f}s')
            update_progress(status='EVALUATING',evaluation_status='RUNNING',current=f'{run}:{last}/{len(ds)}')
        if stopped() and last<len(ds): return False
    del logits,prob
    dump(PRIVATE/f'{stem}_hashes.json',{'logits_sha256':sha(lp),'probabilities_sha256':sha(pp)})
    dump(OUT/f'{stem}_INFERENCE_SUMMARY.json',read(state_path))
    return True

def main():
    protocol=read(OUT/'P4_FROZEN_PROTOCOL.json')
    for item in protocol['manifest']: assert sha(item['path'])==item['sha256'],item['path']
    assert torch.cuda.is_available(),'CUDA unavailable'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    dump(OUT/'INFERENCE_RUNTIME.json',{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'precision':'FP32',
        'TF32':False,'inference_mode':True,'platform':platform.platform(),
        'free_cuda_bytes':torch.cuda.mem_get_info()[0]})
    completed=[]; invocation=time.time()
    for identity in protocol['models']:
        if stopped():
            update_progress(status='PARTIAL',evaluation_status='PARTIAL',evaluation_completed=len(completed)); return
        run=identity['run']; random.seed(identity['training_seed']); np.random.seed(identity['training_seed'])
        torch.manual_seed(identity['training_seed']); torch.cuda.manual_seed_all(identity['training_seed'])
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            ckpt=torch.load(identity['path'],map_location='cpu',weights_only=True)
        assert ckpt['epoch']==identity['best_epoch']
        cfg=copy.deepcopy(ckpt['config']['original_config']); cfg['model']['pretrained']=False
        cfg['training_seed']=identity['training_seed']
        model=(FixedCandidate(cfg,identity['model'],identity['training_seed'])
            if identity['model'] in ['R_IMG','R_CLIN','XATTN'] else OncoDermMambaX(cfg))
        model.load_state_dict(ckpt['model'],strict=True); model.requires_grad_(False).eval().cuda()
        assert all(not m.training for m in model.modules())
        assert all(p.dtype==torch.float32 for p in model.parameters())
        config_dir=OUT/'effective_configs'; config_dir.mkdir(exist_ok=True)
        dump(config_dir/f'{run}.json',cfg)
        dump(OUT/f'{run}_EVALUATION_MODEL.json',{**identity,'strict_load':True,'best_reselected':False,
            'source_model':sys.modules[OncoDermMambaX.__module__].__file__,
            'candidate_source':sys.modules[FixedCandidate.__module__].__file__ if isinstance(model,FixedCandidate) else None,
            'precision':'FP32','TF32':False,'eval_mode':True,'new_training':False})
        del ckpt; gc.collect()
        ds=EvaluationData(cfg); use_cache=verify_cache(model,ds,run)
        if not infer(model,ds,identity,use_cache):
            update_progress(status='PARTIAL',evaluation_status='PARTIAL',evaluation_completed=len(completed)); return
        completed.append(run)
        elapsed=time.time()-invocation
        estimate=elapsed/len(completed)*(15-len(completed))
        dump(OUT/'EVALUATION_STATUS.json',{'status':'RUNNING','completed':completed,'models':15,
            'elapsed_seconds':elapsed,'estimated_inference_seconds_remaining':estimate})
        log(f'MODEL_COMPLETE {run}; models={len(completed)}/15; estimate_remaining={estimate:.1f}s')
        update_progress(evaluation_completed=len(completed),estimated_inference_seconds_remaining=estimate)
        del model,ds; gc.collect(); torch.cuda.empty_cache()
        if time.time()+estimate+1800>DEADLINE and len(completed)<15:
            update_progress(status='PARTIAL',evaluation_status='PARTIAL',reason='Estimated inference plus 30-minute statistics reserve exceeds deadline'); return
    dump(OUT/'EVALUATION_STATUS.json',{'status':'COMPLETE','completed':completed,'models':15,
        'elapsed_seconds':time.time()-invocation,'new_training_runs':0,'cohort':'Derm7pt'})
    update_progress(status='STATISTICS_PENDING',evaluation_status='COMPLETE',evaluation_completed=15)

if __name__=='__main__':
    raise SystemExit('Use p4_entry.py for lightweight spawned workers.')
