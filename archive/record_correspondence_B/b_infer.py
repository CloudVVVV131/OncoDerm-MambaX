"""Fixed six-model clinical score grids; one visual pass per image, no training."""
import os
for key in ["HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"]:
    os.environ[key]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
from b_common import *
import copy
import sys
import time
import traceback
import gc
import numpy as np
import pandas as pd
import torch
import yaml
import torchvision
import timm
import PIL
from b_data import Images,worker_init
from lesionmamba.models.oncoderm_mambax import build_oncoderm_mambax
import clinical_matching_reference as ref

def probabilities(logits):
    return torch.stack([logits[...,:7].softmax(-1)[...,4],logits[...,7].sigmoid()],-1)

def visual(model,x):
    f,a=model._fuse_features(list(model.backbone(x)))
    return model._pool(f,a)

def head(model,f,m):
    e=model.metadata_encoder(m)
    z=f*(1+model.metadata_gate(e).sigmoid())+e
    return torch.cat([model.head(z),model.mel_head(z)],-1)

def grid_logits(model,f,vectors):
    # Original CMGF arithmetic and heads, evaluated for each finite clinical type.
    # Per-type batches match the original head path and avoid large-B GEMM drift.
    return torch.stack([head(model,f,v.expand(len(f),-1)) for v in vectors],1)

def load(entry):
    assert sha(entry["checkpoint_path"])==entry["sha256"]
    assert sha(entry["config_path"])==entry["config_sha256"]
    cfg=yaml.safe_load((OUT/"original_configs"/f'{entry["tag"]}.yaml').read_text(encoding="utf-8-sig"))
    assert sha(OUT/"original_configs"/f'{entry["tag"]}.yaml')==entry["config_sha256"]
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        c=torch.load(entry["checkpoint_path"],map_location="cpu",weights_only=True,mmap=True)
    assert c["config"]["model"]==cfg["model"]
    data=copy.deepcopy(cfg["data"])
    data["metadata"]["metadata_dim"]=22
    assert c["config"]["data"]==data
    assert int(c["epoch"])==entry["original_selected_epoch"]
    ctor=copy.deepcopy(cfg)
    ctor["model"]["pretrained"]=False
    model=build_oncoderm_mambax(ctor)
    keys=model.load_state_dict(c["model"],strict=True)
    assert not keys.missing_keys and not keys.unexpected_keys
    model.requires_grad_(False).eval().cuda()
    assert model.metadata_dim==22 and model.mel_head is not None
    assert not any(m.training for m in model.modules())
    assert all(p.dtype==torch.float32 for p in model.parameters())
    del c
    return model,cfg

def old_scores(entry,cohort,order):
    if cohort=="SIIM":
        records=read(B2/"PREDICTION_MANIFEST.json")
        if isinstance(records,dict):
            records=records.get("predictions",records.get("records",records))
        candidates=[r for r in records if r["dataset"]=="SIIM" and r["model"]==entry["model"].replace("ORIG_","") and int(r["training_seed"])==entry["training_seed"]]
        assert len(candidates)==1
        r=candidates[0]
        assert sha(r["path"])==r["sha256"]
        f=pd.read_csv(r["path"],dtype={"image_id":str})
        ids=[r["image_id"] for r in rows(PRIVATE/cohort/"images.csv")]
        out=[]
        for cond in ["E_REAL","E_NULL"]:
            part=f[f.condition==cond].set_index("image_id").loc[ids]
            assert len(part)==len(ids)
            out.append(part[["prob_mel","prob_mel_endpoint"]].to_numpy())
        return np.stack(out),dict(path=r["path"],sha256=r["sha256"])
    detail=read(OC/f'{entry["tag"]}_INFERENCE.json')
    path=OC/"private_predictions"/f'{entry["tag"]}.npy'
    # The saved inference receipt identifies the exact native/null artifact.
    expected=detail.get("prediction_sha256",detail.get("sha256"))
    assert expected is not None,detail.keys()
    assert sha(path)==expected
    a=np.load(path,allow_pickle=False)
    assert a.shape==(2,len(order),2)
    return a[:,np.argsort(order)],dict(path=str(path),sha256=expected)

def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    env=dict(time=now(),python=sys.version,executable=sys.executable,torch=str(torch.__version__),
             numpy=np.__version__,pandas=pd.__version__,torchvision=torchvision.__version__,
             timm=timm.__version__,pillow=PIL.__version__,GPU=torch.cuda.get_device_name(0),
             precision="FP32",TF32=False,AMP=False,eval=True,inference_mode=True,
             actual_model_source=str(OUT/"model_code/lesionmamba/models/oncoderm_mambax.py"),
             checkpoint_loading="weights_only=True, mmap=True, strict=True",
             readonly_historical_sources=True)
    write(OUT/"ENVIRONMENT.json",env)
    freeze=read(OUT/"B_RUN_PROTOCOL.json")
    for p,h in freeze["model_sources"].items():
        assert sha(OUT/p)==h
    units=[]
    for entry in read(OUT/"FIXED_MODEL_IDENTITIES.json"):
        if expired():break
        todo=[c for c in ["SIIM","Derm7pt"] if not (OUT/f'{entry["tag"]}_{c}_COMPLETE.json').exists()]
        if not todo:continue
        log(f'Strict loading {entry["tag"]}')
        model,cfg=load(entry)
        for cohort in todo:
            if expired():break
            t=time.monotonic()
            p=PRIVATE/cohort
            b=read(p/"binding.json")
            v=torch.from_numpy(np.load(p/"vectors.npy")).cuda()
            types=np.load(p/"types.npy")
            y=np.load(p/"labels.npy")
            order=np.load(p/"old_order.npy")
            n,u=b["n"],b["grid_types"]
            batch=int(entry["B2_HAM_val_prediction_index"]["batch"]) if cohort=="SIIM" else 8
            old,old_identity=old_scores(entry,cohort,order)
            assert old.shape==(2,n,2)
            # Canonical first eight plus every unique vector, including E_NULL.
            sample=Images(cohort,cfg,np.arange(min(8,n)))
            x=torch.stack([sample[i][0] for i in range(len(sample))]).cuda()
            checks=[]
            with torch.inference_mode():
                f=visual(model,x)
                all_logits=grid_logits(model,f,v)
                for j in range(u):
                    m=v[j].expand(len(x),-1)
                    full=model(x,metadata=m)
                    z=torch.cat([full["logits"],full["mel_logits"][:,None]],1)
                    h=head(model,f,m)
                    torch.testing.assert_close(z,h,atol=1e-5,rtol=1e-4)
                    torch.testing.assert_close(z,all_logits[:,j],atol=1e-5,rtol=1e-4)
                    checks.append(dict(type_index=j,max_logit_difference=float((z-all_logits[:,j]).abs().max()),
                                       max_probability_difference=float((probabilities(z)-probabilities(all_logits[:,j])).abs().max())))
            write(OUT/f'{entry["tag"]}_{cohort}_EQUIVALENCE.json',
                  dict(status="PASS",first_canonical_images=len(x),clinical_types=u,checks=checks,
                       atol=1e-5,rtol=1e-4,strict_load=True,epoch=entry["original_selected_epoch"]))
            del x,f,all_logits,z,h,full
            gp=p/f'{entry["tag"]}_score_grid.npy'
            cp=p/f'{entry["tag"]}_grid_checkpoint.json'
            start=0
            if cp.exists():
                saved=read(cp)
                assert saved["checkpoint_sha256"]==entry["sha256"]
                assert saved["order_sha256"]==sha(p/"old_order.npy")
                start=saved["next_position"]
                grid=np.lib.format.open_memmap(gp,mode="r+")
            else:
                grid=np.lib.format.open_memmap(gp,mode="w+",dtype=np.float32,shape=(n,u,2))
            ds=Images(cohort,cfg,order[start:])
            loader=torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=False,num_workers=2,
                pin_memory=True,worker_init_fn=worker_init,persistent_workers=False)
            pos=start
            with torch.inference_mode():
                for x,idx in loader:
                    if expired():break
                    x=x.cuda(non_blocking=True)
                    f=visual(model,x)
                    values=probabilities(grid_logits(model,f,v)).cpu().numpy()
                    ix=idx.numpy()
                    assert np.isfinite(values).all()
                    native=values[np.arange(len(ix)),types[ix]]
                    nul=values[:,b["null_type"]]
                    error=float(np.max(np.abs(np.stack([native,nul])-old[:,ix])))
                    assert error<=2e-6,f"Old FP32 compatibility exceeded: {entry['tag']} {cohort} {error}"
                    grid[ix]=values
                    pos+=len(ix)
                    if pos% (batch*100)==0 or pos==n:
                        grid.flush()
                        write(cp,dict(next_position=pos,n=n,checkpoint_sha256=entry["sha256"],
                                      order_sha256=sha(p/"old_order.npy"),time=now()))
                        log(f'{entry["tag"]} {cohort}: {pos}/{n}')
            grid.flush()
            write(cp,dict(next_position=pos,n=n,checkpoint_sha256=entry["sha256"],order_sha256=sha(p/"old_order.npy"),time=now()))
            if pos<n:
                log("Window deadline: inference checkpoint saved")
                break
            new=np.stack([grid[np.arange(n),types],grid[:,b["null_type"]]])
            checks_metrics=[]
            for cond in range(2):
                for head_index in range(2):
                    a=ref.ap_auc(y,new[cond,:,head_index])
                    z=ref.ap_auc(y,old[cond,:,head_index])
                    diff=float(np.max(np.abs(a-z)))
                    assert diff<=5e-6
                    checks_metrics.append(dict(condition=cond,head=head_index,max_AP_AUC_difference=diff))
            detail=dict(status="COMPLETE",tag=entry["tag"],cohort=cohort,n=n,u=u,grid_shape=list(grid.shape),
                        grid_sha256=sha(gp),checkpoint_sha256=entry["sha256"],selected_epoch=entry["original_selected_epoch"],
                        seconds=time.monotonic()-t,batch=batch,old_prediction=old_identity,
                        max_old_probability_difference=float(np.max(np.abs(new-old))),metrics_compatibility=checks_metrics,
                        visual_calls="one per scored image plus explicitly bounded equivalence checks",
                        completed_at=now())
            write(OUT/f'{entry["tag"]}_{cohort}_COMPLETE.json',detail)
            units.append(detail)
            log(f'{entry["tag"]} {cohort} COMPLETE in {detail["seconds"]:.1f}s')
            del grid,old,new,loader,ds,v
        del model
        gc.collect()
        torch.cuda.empty_cache()
    log("Inference stage exited")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        write(OUT/"INFERENCE_BLOCKER.json",dict(time=now(),error=str(e),traceback=traceback.format_exc()))
        raise
