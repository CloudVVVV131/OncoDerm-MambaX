"""Original B2/B6 only. Restricted loading and verified paired feature reuse."""
import copy
import gc
import platform
import sys
import time
import numpy as np
import torch
import torchvision
import timm
import yaml
from torch.utils.data import DataLoader, Subset
from core_common import *
from core_data import DermData, worker_init
from lesionmamba.models.oncoderm_mambax import build_oncoderm_mambax
import lesionmamba.models.oncoderm_mambax as actual_module

def full(model,x,m):
    out=model(x,metadata=m)
    return torch.cat([out['logits'],out['mel_logits'].reshape(-1,1)],1)

def visual(model,x):
    features,attention=model._fuse_features(list(model.backbone(x)))
    return model._pool(features,attention)

def head(model,features,m):
    encoded=model.metadata_encoder(m)
    fused=features*(1+torch.sigmoid(model.metadata_gate(encoded)))+encoded
    return torch.cat([model.head(fused),model.mel_head(fused)],1)

def load(entry):
    assert sha(entry['checkpoint_path'])==entry['sha256']
    assert sha(entry['config_path'])==entry['config_sha256']
    cfg=yaml.safe_load(Path(entry['config_copy']).read_text(encoding='utf-8-sig'))
    assert sha(entry['config_copy'])==entry['config_sha256']
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ckpt=torch.load(entry['checkpoint_path'],map_location='cpu',weights_only=True,mmap=True)
    assert ckpt['config']['model']==cfg['model']
    data=copy.deepcopy(cfg['data']);data['metadata']['metadata_dim']=22
    assert ckpt['config']['data']==data
    assert int(ckpt['epoch'])==int(entry['original_selected_epoch'])
    construct=copy.deepcopy(cfg);construct['model']['pretrained']=False
    model=build_oncoderm_mambax(construct)
    keys=model.load_state_dict(ckpt['model'],strict=True)
    assert not keys.missing_keys and not keys.unexpected_keys
    assert model.metadata_dim==22 and model.mel_head is not None
    model.requires_grad_(False).eval().cuda()
    assert not any(m.training for m in model.modules())
    assert all(p.dtype==torch.float32 for p in model.parameters())
    detail=dict(tag=entry['tag'],run_id=entry['run_id'],checkpoint_sha256=entry['sha256'],epoch=int(ckpt['epoch']),
        strict=True,missing_keys=[],unexpected_keys=[],source_path=actual_module.__file__,source_sha256=sha(actual_module.__file__),
        safe_load='weights_only=True; only TorchVersion explicitly allowlisted',constructor_override='pretrained=False only; full checkpoint strict restore',
        config_check='identical model; data identical after documented derived metadata_dim22',metadata=model.metadata)
    del ckpt
    return model,cfg,detail

def infer_one(entry):
    tag=entry['tag'];path=PRIVATE/(tag+'.npy');status=OUT/(tag+'_INFERENCE.json')
    if status.exists() and read(status)['status']=='COMPLETE':
        old=read(status);assert sha(path)==old['prediction_sha256'];log('Reuse completed '+tag);return old
    start=time.perf_counter();model,cfg,identity=load(entry)
    ds=DermData(cfg);n=len(ds)
    with torch.inference_mode():
        first=[ds[i] for i in range(min(8,n))]
        x=torch.stack([a[0] for a in first]).cuda()
        m=torch.stack([a[1] for a in first]).cuda()
        null=ds.null.cuda().expand(len(x),-1)
        z=[full(model,x,m),full(model,x,null)]
        feat=visual(model,x);cached=[head(model,feat,m),head(model,feat,null)]
        passed=all(torch.allclose(a,b,atol=1e-5,rtol=1e-4) for a,b in zip(z,cached))
        error=max(float((a-b).abs().max()) for a,b in zip(z,cached))
        assert all(a.shape==(len(x),8) and torch.isfinite(a).all() for a in z+cached)
    identity.update(cache_passed=passed,max_logit_abs_error=error,cache_atol=1e-5,cache_rtol=1e-4,technical_images=len(x),
        clinical_encoder_exact_match=True,E_NULL_original_encoder_exact=True,transform=repr(ds.transform),batch=8,workers=2)
    dump(OUT/(tag+'_TECHNICAL.json'),identity)
    del x,m,null,z,feat,cached,first
    resume=PRIVATE/(tag+'_checkpoint.json')
    done=0
    if resume.exists():
        state=read(resume);assert state['checkpoint_sha256']==entry['sha256'] and state['protocol_sha256']==sha(OUT/'ORIGINALCORE_PROTOCOL.json')
        done=state['completed_rows']
        scores=np.lib.format.open_memmap(path,mode='r+')
    else:
        scores=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(2,n,2))
        scores[:]=np.nan;scores.flush()
    dl=DataLoader(Subset(ds,range(done,n)),batch_size=8,shuffle=False,num_workers=2,pin_memory=True,worker_init_fn=worker_init,persistent_workers=False)
    with torch.inference_mode():
        for x,m,ix in dl:
            x=x.cuda(non_blocking=True);m=m.cuda(non_blocking=True)
            null=ds.null.cuda().expand(len(x),-1)
            if passed:
                feat=visual(model,x);values=[head(model,feat,m),head(model,feat,null)]
            else: values=[full(model,x,m),full(model,x,null)]
            for c,v in enumerate(values):
                assert v.shape==(len(x),8) and torch.isfinite(v).all()
                scores[c,ix.numpy(),0]=v[:,:7].softmax(1)[:,4].cpu().numpy().astype(np.float64)
                scores[c,ix.numpy(),1]=v[:,7].sigmoid().cpu().numpy().astype(np.float64)
            done=int(ix[-1])+1
            scores.flush()
            dump(resume,dict(completed_rows=done,checkpoint_sha256=entry['sha256'],protocol_sha256=sha(OUT/'ORIGINALCORE_PROTOCOL.json'),updated_at=now()))
            if done%256<8 or done==n:log(f'{tag} {done}/{n} elapsed={time.perf_counter()-start:.1f}s')
            if time.time()>=DEADLINE-180:break
    del dl
    assert np.isfinite(scores[:,:done,:]).all()
    scores.flush();del scores
    complete=done==n
    result=dict(status='COMPLETE' if complete else 'PARTIAL',tag=tag,run_id=entry['run_id'],checkpoint_sha256=entry['sha256'],
       rows=done,expected_rows=n,score_vectors=4 if complete else 0,conditions=CONDITIONS,heads=HEADS,
       prediction_path=str(path),prediction_sha256=sha(path),seconds=time.perf_counter()-start,finished_at=now(),cache_reuse=passed)
    dump(status,result)
    del model,ds;gc.collect();torch.cuda.empty_cache()
    return result

def main():
    protocol=read(OUT/'ORIGINALCORE_PROTOCOL.json')
    assert protocol['usage_history']['user_statement']=='确认六个原模型及其开发过程均未主动使用 Derm7pt'
    assert torch.cuda.is_available(),'CUDA unavailable'
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    assert Path(actual_module.__file__).is_relative_to(OUT/'model_code')
    for r in table(OUT/'MODEL_SOURCE_MANIFEST.csv'):assert sha(r['actual_import_copy'])==r['sha256']
    dump(OUT/'INFERENCE_ENVIRONMENT.json',dict(time=now(),python=sys.version,executable=sys.executable,torch=str(torch.__version__),
        torchvision=torchvision.__version__,timm=timm.__version__,numpy=np.__version__,device=torch.cuda.get_device_name(0),
        cuda_runtime=torch.version.cuda,total_vram=torch.cuda.get_device_properties(0).total_memory,precision='FP32',TF32=False,
        disk_free=__import__('shutil').disk_usage(OUT).free))
    progress(status='RUNNING',inference='RUNNING')
    records=[]
    for entry in protocol['models']:
        if time.time()>=DEADLINE-180:break
        record=infer_one(entry);records.append(record)
        progress(completed_models=sum(r['status']=='COMPLETE' for r in records),identity='STRICT_LOAD_'+str(len(records))+'_OF_6')
        if len(records)==1:
            estimate=record['seconds']*5+900
            log(f'First model done. Remaining inference/statistics estimate={estimate/60:.1f}min; remaining window={(DEADLINE-time.time())/60:.1f}min')
        if record['status']!='COMPLETE':break
    complete=len(records)==6 and all(r['status']=='COMPLETE' for r in records)
    dump(OUT/'INFERENCE_SUMMARY.json',dict(status='COMPLETE' if complete else 'PARTIAL',records=records,finished_at=now()))
    progress(status='INFERENCE_COMPLETE' if complete else 'PARTIAL',inference='COMPLETE' if complete else 'PARTIAL',identity='PASS' if complete else 'PARTIAL')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        progress(status='BLOCKED',inference='BLOCKED',error=repr(exc));raise
