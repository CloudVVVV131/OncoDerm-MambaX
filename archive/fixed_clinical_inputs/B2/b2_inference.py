"""B2 fixed-checkpoint FP32 paired inference, 2026-10-12.
No training, cloud, downloads, arbitrary pickle, or historical driver execution.
Only copied exact model modules are imported. Seed for technical repeatability:42.
"""
from b2_common import *
import argparse
import copy
import gc
import platform
import time
import numpy as np
import pandas as pd
import torch
import torchvision
import timm
import PIL
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0,str(OUT/'model_code'))
from lesionmamba.models.oncoderm_mambax import build_oncoderm_mambax
from lesionmamba.data.datasets import encode_metadata, LABEL_MAP
from lesionmamba.data.transforms import build_transforms

NULL=torch.tensor([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1],dtype=torch.float32)


def setup():
    assert torch.cuda.is_available(), 'CUDA unavailable; CPU inference prohibited'
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    assert LABEL_MAP['mel']==4
    return dict(time=now(),python=sys.version,executable=sys.executable,platform=platform.platform(),
        torch=str(torch.__version__),torchvision=torchvision.__version__,timm=timm.__version__,
        pillow=PIL.__version__,numpy=np.__version__,pandas=pd.__version__,CUDA_runtime=torch.version.cuda,
        device=torch.cuda.get_device_name(0),device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        precision='FP32',TF32=False,autocast=False,eval=True,inference_mode=True,threads=4,
        safe_loading='weights_only=True; explicit allowlist only torch.torch_version.TorchVersion; no unrestricted pickle',
        network_policy='no download; offline flags; constructor pretrained=False, then full strict state_dict restore')


class Images(Dataset):
    def __init__(self,name,config,limit=None):
        self.name=name
        records=csv_rows(LOCAL/f'metadata/{name}_images.csv')
        if name=='SIIM':
            self.meta=pd.read_csv(LOCAL/'metadata/SIIM_alignment.csv',dtype={'image_name':str,'patient_id':str})
            self.ids=self.meta.image_name.tolist()
            self.clinical=np.load(LOCAL/'metadata/SIIM_E_REAL.npy',allow_pickle=False)
        else:
            split=name.split('_')[1]
            self.meta=pd.read_csv(LOCAL/f'metadata/HAM_{split}.csv',dtype={'image_id':str})
            self.ids=self.meta.image_id.tolist()
            self.clinical=np.stack([encode_metadata(row,config['data']['metadata']).numpy() for _,row in self.meta.iterrows()])
        assert self.ids==[r['image_id'] for r in records]
        self.paths=[r['path'] for r in records]
        self.records=records
        self.transform=build_transforms(config,'eval')
        self.count=len(records) if limit is None else min(limit,len(records))

    def __len__(self):return self.count

    def __getitem__(self,index):
        row=self.records[index]
        p=Path(row['path']); s=p.stat()
        assert s.st_size==int(row['bytes']) and s.st_mtime_ns==int(row['mtime_ns']), 'Image changed '+str(p)
        # Hash the actual source image on its first technical/full encounter in each unit.
        assert sha(p)==row['sha256_historical'], 'Image content hash differs '+str(p)
        with Image.open(p) as image:
            image.load()
            tensor=self.transform(image.convert('RGB'))
        return tensor, torch.from_numpy(self.clinical[index].copy())


def load(entry):
    cfg=yaml.safe_load(Path(entry['config_path']).read_text(encoding='utf-8-sig'))
    assert sha(entry['checkpoint_path'])==entry['historical_checkpoint_sha256_NOT_recomputed']
    assert sha(entry['config_path'])==entry['config_sha256_now']
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ckpt=torch.load(entry['checkpoint_path'],map_location='cpu',weights_only=True,mmap=True)
    assert isinstance(ckpt,dict) and isinstance(ckpt['model'],dict)
    assert ckpt['config']['model']==cfg['model'], 'Checkpoint/config model mismatch'
    normalized_data=copy.deepcopy(cfg['data'])
    normalized_data['metadata']['metadata_dim']=22
    assert ckpt['config']['data']==normalized_data, 'Checkpoint/config data mismatch beyond derived metadata_dim=22'
    construct=copy.deepcopy(cfg);construct['model']['pretrained']=False
    model=build_oncoderm_mambax(construct)
    keys=model.load_state_dict(ckpt['model'],strict=True)
    assert not keys.missing_keys and not keys.unexpected_keys
    assert model.metadata_dim==22 and model.metadata_encoder is not None and model.mel_head is not None
    model.requires_grad_(False).eval().cuda()
    assert not any(m.training for m in model.modules())
    detail=dict(run_id=entry['run_id'],model=entry['model'],seed=int(entry['training_seed']),epoch=ckpt['epoch'],
        checkpoint_sha256=sha(entry['checkpoint_path']),config_sha256=sha(entry['config_path']),
        strict=True,missing_keys=[],unexpected_keys=[],metadata_dim=22,MEL_class_index=4,
        checkpoint_model_config_matches=True,data_config_match_after_derived_metadata_dim22=True,
        data_config_difference='Checkpoint stores derived metadata_dim=22; sidecar omits it. All other data fields equal.',parameter_dtype=str(next(model.parameters()).dtype),
        source='model_code/lesionmamba/models/oncoderm_mambax.py',prefix_conversion='none',
        constructor_only_override='model.pretrained=False before strict restore',pretrained_download_disabled=True)
    del ckpt
    return model,cfg,detail


def full(model,x,m):
    value=model(x,metadata=m)
    return torch.cat([value['logits'],value['mel_logits'].reshape(-1,1)],dim=1)


def visual(model,x):
    feat,att=model._fuse_features(list(model.backbone(x)))
    return model._pool(feat,att)


def head(model,feat,m):
    encoded=model.metadata_encoder(m)
    fused=feat*(1+torch.sigmoid(model.metadata_gate(encoded)))+encoded
    return torch.cat([model.head(fused),model.mel_head(fused)],dim=1)


def forward_conditions(model,x,m,paired,reuse):
    if reuse:
        feat=visual(model,x)
        return ([head(model,feat,NULL.to(x.device).expand(len(x),-1)),head(model,feat,m)] if paired else [head(model,feat,m)])
    return ([full(model,x,NULL.to(x.device).expand(len(x),-1)),full(model,x,m)] if paired else [full(model,x,m)])


def loader(dataset,batch):
    # No persistent worker/helper processes. Four CPU torch threads, one reader.
    return DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0,pin_memory=True)


def technical():
    guard_sources()
    env=setup()
    if not (OUT/'ENVIRONMENT.json').exists():
        save_json('ENVIRONMENT.json',env)
    else:
        for key in ['executable','torch','torchvision','timm','CUDA_runtime','device','precision']:
            assert env[key]==js(OUT/'ENVIRONMENT.json')[key]
    details=[]
    for entry in entries():
        tag=model_tag(entry); log('Strict load and HAM-validation technical check: '+tag)
        model,cfg,detail=load(entry)
        ds=Images('HAM_val',cfg,96)
        torch.cuda.reset_peak_memory_stats()
        first=next(iter(loader(ds,8)))
        x,m=[t.cuda(non_blocking=True) for t in first]
        seen=[]
        hook=model.metadata_encoder.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().cpu().numpy().copy()))
        with torch.inference_mode():
            null=NULL.cuda().expand(len(x),-1)
            znull=full(model,x,null);zreal=full(model,x,m)
            assert np.array_equal(seen[0],null.cpu().numpy()) and np.array_equal(seen[1],m.cpu().numpy())
            hook.remove()
            features=visual(model,x)
            cnull=head(model,features,null);creal=head(model,features,m)
            good=bool(torch.allclose(cnull,znull,atol=1e-5,rtol=1e-4) and torch.allclose(creal,zreal,atol=1e-5,rtol=1e-4))
            for z in [znull,zreal,cnull,creal]:
                assert z.shape==(8,8) and torch.isfinite(z).all()
                p=z[:,:7].softmax(1); ep=z[:,7].sigmoid()
                assert torch.allclose(p.sum(1),torch.ones(8,device='cuda'),atol=1e-6)
                assert torch.isfinite(ep).all() and (ep>=0).all() and (ep<=1).all()
            error=float(max((cnull-znull).abs().max().item(),(creal-zreal).abs().max().item()))
        detail.update(cache_equivalence_pass=good,cache_max_logit_error=error,metadata_branch_verified=True,
            logits_finite=True,probability_conversion_pass=True,transform_repr=repr(ds.transform),
            technical_ids_selection='First96 lexicographic fixed original HAM validation IDs, not outcome-selected')
        trials=[]
        for batch in [8,16,32]:
            try:
                gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize();start=time.perf_counter();n=0
                with torch.inference_mode():
                    for x,m in loader(ds,batch):
                        x=x.cuda(non_blocking=True);m=m.cuda(non_blocking=True)
                        values=forward_conditions(model,x,m,True,good)
                        assert all(torch.isfinite(z).all() for z in values)
                        # Include output transfer, image hash, decode, preprocessing and clinical transfer.
                        for z in values:_=z.cpu().numpy()
                        n+=len(x)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                trials.append(dict(batch=batch,n=n,seconds=elapsed,images_per_second=n/elapsed,
                    peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),status='PASS'))
            except torch.cuda.OutOfMemoryError:
                trials.append(dict(batch=batch,status='OOM'))
                gc.collect();torch.cuda.empty_cache()
                break
        valid=[r for r in trials if r['status']=='PASS']
        assert valid,'No viable batch '+tag
        choice=max(valid,key=lambda r:r['images_per_second'])
        # Conservative validation-derived allowance for external JPEG IO and normal load variation.
        seconds=(33126+1513+1520)/choice['images_per_second']*1.5
        detail.update(trials=trials,batch=choice['batch'],reuse_features=good,
            selected_throughput=choice['images_per_second'],estimated_inference_seconds_with_1_5_margin=seconds)
        save_json('technical/'+tag+'.json',detail);details.append(detail)
        log(tag+f': batch={choice["batch"]}, {choice["images_per_second"]:.2f} images/s, cache_error={error:.3g}, projected {seconds/60:.1f} min')
        del model,ds,x,m;gc.collect();torch.cuda.empty_cache()
    expected=sum(r['estimated_inference_seconds_with_1_5_margin'] for r in details)+1800
    status='PASS' if expected<=14400 else 'PAUSED_ESTIMATE_OVER_4_HOURS'
    save_json('TECHNICAL_CHECKS.json',dict(time=now(),status=status,models=details,
        device=env['device'],statistical_allowance_seconds=1800,total_projected_seconds=expected,
        throughput_scope='Image read/hash/decode/preprocess + paired forwards + output copy on fixed HAM validation sample',
        uncertainty='Validation-based estimate with1.5x inference safety factor; runtime deadline also enforced'))
    log(f'Technical gate: {status}; whole-round projection {expected/3600:.2f} hours')


def infer():
    gate=js(OUT/'TECHNICAL_CHECKS.json')
    assert gate['status']=='PASS','Technical/time gate not passed'
    guard_sources();setup()
    start=time.perf_counter();deadline=14400-gate['statistical_allowance_seconds']
    records=[]
    for entry,settings in zip(entries(),gate['models']):
        assert entry['run_id']==settings['run_id']
        tag=model_tag(entry)
        model,cfg,detail=load(entry)
        for name in ['HAM_val','HAM_test','SIIM']:
            done=OUT/f'local_only/predictions/{tag}_{name}.json'
            if done.exists():
                saved=js(done)
                assert sha(saved['path'])==saved['sha256']
                records.append(saved);log('Reuse verified completed unit '+tag+' '+name);continue
            ds=Images(name,cfg)
            conditions=['E_NULL','E_REAL'] if name=='SIIM' else ['NATIVE']
            buffers=[[] for _ in conditions];probabilities=[[] for _ in conditions];endpoint=[[] for _ in conditions]
            unit_start=time.perf_counter();n=0
            with torch.inference_mode():
                for x,m in loader(ds,settings['batch']):
                    x=x.cuda(non_blocking=True);m=m.cuda(non_blocking=True)
                    values=forward_conditions(model,x,m,name=='SIIM',settings['reuse_features'])
                    for i,z in enumerate(values):
                        assert z.shape[1]==8 and torch.isfinite(z).all()
                        buffers[i].append(z.cpu().numpy());probabilities[i].append(z[:,:7].softmax(1).cpu().numpy());endpoint[i].append(z[:,7].sigmoid().cpu().numpy())
                    n+=len(x)
                    if n%2048<settings['batch'] or n==len(ds):
                        log(f'{tag} {name}: {n}/{len(ds)}; elapsed {time.perf_counter()-start:.1f}s')
                    if time.perf_counter()-start>deadline:
                        save_json('INFERENCE_PAUSED_'+str(time.time_ns())+'.json',dict(status='PARTIAL_TIME_LIMIT',time=now(),
                            completed_units=records,current_incomplete_unit=tag+'_'+name,elapsed=time.perf_counter()-start,
                            reason='Reserved remaining window for statistics/report; incomplete unit not reported as completed'))
                        return
            frames=[]
            ids=ds.ids
            label=ds.meta.target.to_numpy() if name=='SIIM' else (ds.meta.dx=='mel').to_numpy(dtype=int)
            for i,condition in enumerate(conditions):
                z=np.concatenate(buffers[i]);p=np.concatenate(probabilities[i]);ep=np.concatenate(endpoint[i])
                assert len(z)==len(ds) and np.isfinite(z).all()
                frame=pd.DataFrame(dict(image_id=ids,label=label,run_id=entry['run_id'],model=entry['model'],training_seed=int(entry['training_seed']),dataset=name,condition=condition,prob_mel=p[:,4],prob_mel_endpoint=ep,mel_logit=z[:,7]))
                for k in range(7):frame[f'logit_{k}']=z[:,k];frame[f'prob_{k}']=p[:,k]
                frames.append(frame)
            path=OUT/f'local_only/predictions/{tag}_{name}.csv'
            pd.concat(frames,ignore_index=True).to_csv(path,index=False,mode='x',float_format='%.17g')
            record=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,run_id=entry['run_id'],
                model=entry['model'],training_seed=int(entry['training_seed']),dataset=name,conditions=conditions,
                images=len(ds),prediction_rows=sum(len(f) for f in frames),batch=settings['batch'],
                checkpoint_sha256=detail['checkpoint_sha256'],precision='FP32_TF32_off',reuse_features=settings['reuse_features'],
                image_hash_and_decode_checked=len(ds),seconds=time.perf_counter()-unit_start,completed_at=now())
            save_json(str(done.relative_to(OUT)),record);records.append(record)
            log('Saved verified unit '+tag+' '+name)
        del model;gc.collect();torch.cuda.empty_cache()
    save_json('PREDICTION_MANIFEST.json',records)
    save_json('INFERENCE_STATUS.json',dict(status='COMPLETE',time=now(),units=18,external_condition_units=12,
        HAM_NATIVE_units=12,elapsed_seconds=time.perf_counter()-start,images_per_model=36159,
        weights_loaded_strict=6,inference=True,training=False))
    log('All six models and paired external conditions completed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['technical','infer']);args=parser.parse_args()
    try:
        if args.mode=='technical':technical()
        else:infer()
    except Exception as exc:
        save_json('BLOCKED_'+args.mode+'_'+str(time.time_ns())+'.json',dict(status='BLOCKED',time=now(),error=repr(exc)))
        log('BLOCKED: '+repr(exc));raise
