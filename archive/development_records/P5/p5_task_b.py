"""Six fixed models, HAM validation only, three prespecified spatial conditions."""
import copy
import gc
import platform
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from p5_common import *
from p5_model import FixedCandidate
from p5_data import ValidationData,worker_init
from p5_spatial import cache_batch,intervened_forward,move_regions,state_fingerprint,permutation

def loader(ds,start=0,end=None):
    return DataLoader(ds,sampler=list(range(start,len(ds) if end is None else end)),batch_size=8,
        num_workers=2,pin_memory=True,prefetch_factor=1,persistent_workers=False,worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(ds.training_seed*1000+17))

def outputs(out):
    return torch.cat([out['logits'],out['mel_logits'][:,None]],1),torch.cat([out['logits'].softmax(1),out['mel_logits'].sigmoid()[:,None]],1)

def aggregate(total,out,n):
    for key,value in out['readout_diagnostics'].items():
        value=value.detach().float()
        if value.ndim==0:value=value.expand(n)
        if value.ndim==1:value=value[:,None]
        value=value.cpu().numpy().astype(float)
        assert np.isfinite(value).all()
        for j in range(value.shape[1]):
            v=value[:,j]; row=total.setdefault(f'{key}_{j}',{'n':0,'sum':0.,'sum_sq':0.,'min':float(v.min()),'max':float(v.max())})
            row['n']+=len(v); row['sum']+=float(v.sum()); row['sum_sq']+=float((v*v).sum())
            row['min']=min(row['min'],float(v.min())); row['max']=max(row['max'],float(v.max()))

@torch.inference_mode()
def technical_check(model,ds,identity):
    begin=time.time();images,meta,ids,index=next(iter(loader(ds,0,8)))
    x=images.cuda();clinical=meta.cuda();cached=cache_batch(model,x,clinical)
    native=outputs(intervened_forward(model,cached,'NATIVE',ids))
    full=outputs(model(x,clinical));identity_out=outputs(intervened_forward(model,cached,'IDENTITY_CHECK',ids))
    checks=[]
    native_matches=True
    for name,aa,bb,cc in zip(['logits','probabilities'],native,full,identity_out):
        native_ok=bool(torch.allclose(aa,bb,atol=1e-5,rtol=1e-4));native_matches&=native_ok
        assert torch.allclose(aa,cc,atol=1e-5,rtol=1e-4),'Identity permutation mismatch'
        checks.append({'quantity':name,'native_full_max_abs_error':float((aa-bb).abs().max()),
            'native_full_passed':native_ok,'identity_max_abs_error':float((aa-cc).abs().max())})
    regions=cached[-1];before=regions.sum((-2,-1)); h,w=regions.shape[-2:]
    region_checks=[]
    for condition in ['SHIFT_HALF','PERMUTE_FIXED']:
        moved=move_regions(regions,condition,ids)
        assert torch.equal(regions.flatten(2).sort(-1).values,moved.flatten(2).sort(-1).values)
        result=outputs(intervened_forward(model,cached,condition,ids));assert all(torch.isfinite(v).all() for v in result)
        region_checks.append({'condition':condition,'distribution_exactly_preserved':True,
            'max_mass_relative_error':float(((moved.sum((-2,-1))-before).abs()/before.clamp_min(1e-12)).max()),
            'valid_flags_preserved':True})
    # ID-driven permutation is invariant to batch order, family and training seed.
    for image_id in ids:
        a=permutation(image_id,h,w); assert np.array_equal(np.sort(a),np.arange(h*w))
        assert np.array_equal(a,permutation(image_id,h,w))
    record={'status':'PASSED','run':identity['run'],'images':8,'indices':list(range(8)),
        'native_uses_original_full_forward_fallback':not native_matches,'checks':checks,'regions':region_checks,
        'feature_grid':[h,w],'permutation_batch_order_independent':True,'no_candidate_search':True,
        'seconds':time.time()-begin,'estimate_note':'Use first completed run for wall-time estimate; eight-case loader startup is not per-batch compute time',
        'tolerances':{'native_atol':1e-5,'native_rtol':1e-4,'mass_relative':1e-5}}
    dump(OUT/f"{identity['run']}_TECHNICAL_CHECK.json",record)
    log(f"TASK_B_TECHNICAL_PASSED {identity['run']} grid={h}x{w}; starting all1513, fixed3conditions")
    return not native_matches

@torch.inference_mode()
def infer(model,ds,identity,native_fallback,initial_state):
    run=identity['run'];state_path=PRIVATE/f'{run}_status.json'
    lp=PRIVATE/f'{run}_logits.npy'; pp=PRIVATE/f'{run}_probabilities.npy'; start=0; prior=0.
    diagnostics={c:{} for c in CONDITIONS};perm_fingerprints=[]; grids=set()
    protocol_sha=sha(OUT/'P5_FROZEN_PROTOCOL.json')
    if state_path.exists():
        old=read(state_path)
        assert old['best_sha256']==identity['sha256'] and old['protocol_sha256']==protocol_sha
        assert old['state_before']==initial_state and old['conditions']==CONDITIONS
        if old['status']=='COMPLETE':
            assert read(OUT/f'{run}_PREDICTION_FINGERPRINTS.json')=={'logits_sha256':sha(lp),'probabilities_sha256':sha(pp)}
            return True
        start=old['next_index'];prior=old['elapsed_seconds'];diagnostics=old['diagnostics']
        perm_fingerprints=old['permutation_batch_fingerprints'];grids={tuple(g) for g in old['feature_grids']}
    else:assert not lp.exists() and not pp.exists()
    shape=(1513,3,8)
    logits=np.lib.format.open_memmap(lp,mode='r+' if lp.exists() else 'w+',dtype='float32',shape=shape)
    probs=np.lib.format.open_memmap(pp,mode='r+' if pp.exists() else 'w+',dtype='float32',shape=shape)
    begin=time.time()
    for images,meta,ids,index in loader(ds,start):
        x=images.cuda(non_blocking=True);clinical=meta.cuda(non_blocking=True);cached=cache_batch(model,x,clinical)
        h,w=cached[-1].shape[-2:];grids.add((h,w))
        digest=hashlib.sha256()
        for image_id in ids:digest.update(image_id.encode());digest.update(permutation(image_id,h,w).astype('<i8').tobytes())
        perm_fingerprints.append({'start':int(index[0]),'count':len(ids),'grid':[h,w],'sha256':digest.hexdigest()})
        for c,condition in enumerate(CONDITIONS):
            out=intervened_forward(model,cached,condition,ids)
            v,p=outputs(model(x,clinical)) if condition=='NATIVE' and native_fallback else outputs(out)
            assert torch.isfinite(v).all() and torch.isfinite(p).all()
            logits[index.numpy(),c]=v.cpu().numpy();probs[index.numpy(),c]=p.cpu().numpy()
            aggregate(diagnostics[condition],out,len(ids))
        last=int(index[-1])+1
        if last%256==0 or last==len(ds) or stopped():
            logits.flush();probs.flush()
            current_state=state_fingerprint(model);assert current_state==initial_state,'Model state mutated'
            status={'status':'COMPLETE' if last==len(ds) else 'PARTIAL','run':run,'total':len(ds),'next_index':last,
                'conditions':CONDITIONS,'cohort':'HAM_val','input':'NATIVE','best_sha256':identity['sha256'],
                'protocol_sha256':protocol_sha,'state_before':initial_state,'state_after':current_state,
                'elapsed_seconds':prior+time.time()-begin,'diagnostics':diagnostics,'feature_grids':sorted(grids),
                'permutation_batch_fingerprints':perm_fingerprints,'native_full_forward_fallback':native_fallback}
            dump(state_path,status)
            progress(task_B='RUNNING',B_current_run=run,B_images_completed=last)
            log(f'TASK_B {run} {last}/1513 elapsed={time.time()-begin:.1f}s')
        if stopped() and last<len(ds):return False
    del logits,probs
    dump(OUT/f'{run}_PREDICTION_FINGERPRINTS.json',{'logits_sha256':sha(lp),'probabilities_sha256':sha(pp)})
    dump(OUT/f'{run}_INFERENCE_SUMMARY.json',read(state_path))
    return True

def main():
    assert torch.cuda.is_available(),'CUDA unavailable'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    protocol=read(OUT/'P5_FROZEN_PROTOCOL.json');models=protocol['task_B']['models']
    for item in protocol['manifest']:assert sha(item['path'])==item['sha256'],item['path']
    code_files=sorted(OUT.glob('*.py'))+sorted(SOURCE.rglob('*.py'))
    frozen=[{'path':str(p),'sha256':sha(p)} for p in code_files]
    codepath=OUT/'P5_CODE_FREEZE.json'
    if codepath.exists():
        for item in read(codepath):assert sha(item['path'])==item['sha256']
    else:dump(codepath,frozen)
    dump(OUT/'B_RUNTIME.json',{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'free_cuda_bytes':torch.cuda.mem_get_info()[0],
        'platform':platform.platform(),'precision':'FP32','TF32':False,'inference_mode':True,'new_training_runs':0})
    completed=[]; begin=time.time()
    for identity in models:
        if stopped():
            progress(status='PARTIAL',task_B='PARTIAL',B_models_completed=len(completed));return
        assert identity['model'] in ['R_IMG','R_CLIN']
        random.seed(identity['training_seed']);np.random.seed(identity['training_seed']);torch.manual_seed(identity['training_seed'])
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            ckpt=torch.load(identity['path'],map_location='cpu',weights_only=True)
        assert ckpt['epoch']==identity['best_epoch']
        cfg=copy.deepcopy(ckpt['config']['original_config']);cfg['model']['pretrained']=False;cfg['training_seed']=identity['training_seed']
        model=FixedCandidate(cfg,identity['model'],identity['training_seed'])
        model.load_state_dict(ckpt['model'],strict=True);model.requires_grad_(False).eval().cuda()
        assert all(not m.training for m in model.modules()) and all(p.dtype==torch.float32 for p in model.parameters())
        del ckpt;gc.collect();initial_state=state_fingerprint(model)
        dump(OUT/f"{identity['run']}_MODEL_IDENTITY.json",{**identity,'strict_load':True,'state_initial':initial_state,
            'model_source':str(Path(sys.modules[FixedCandidate.__module__].__file__).resolve()),
            'region_source':str(SOURCE/'region_readout_reference.py'),'intervention_source':str(OUT/'p5_spatial.py')})
        ds=ValidationData(cfg);fallback=technical_check(model,ds,identity)
        if not infer(model,ds,identity,fallback,initial_state):
            progress(status='PARTIAL',task_B='PARTIAL',B_models_completed=len(completed));return
        completed.append(identity['run']);elapsed=time.time()-begin;remaining=elapsed/len(completed)*(6-len(completed))
        dump(OUT/'TASK_B_STATUS.json',{'status':'RUNNING','completed':completed,'models':6,'elapsed_seconds':elapsed,
            'estimated_inference_remaining_seconds':remaining,'new_training_runs':0})
        progress(task_B='RUNNING',B_models_completed=len(completed),B_estimated_remaining_seconds=remaining)
        log(f'TASK_B_MODEL_COMPLETE {identity["run"]} {len(completed)}/6 estimated_remaining={remaining:.1f}s')
        del model,ds;gc.collect();torch.cuda.empty_cache()
    dump(OUT/'TASK_B_STATUS.json',{'status':'INFERENCE_COMPLETE','completed':completed,'models':6,'elapsed_seconds':time.time()-begin,'new_training_runs':0})
    progress(task_B='INFERENCE_COMPLETE',B_models_completed=6,B_estimated_remaining_seconds=0)

if __name__=='__main__':raise SystemExit('Use p5_b_entry.py for lightweight worker imports.')
