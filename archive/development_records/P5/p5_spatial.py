"""Spatial correspondence intervention after soft-region construction only."""
from contextlib import contextmanager
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from p5_model import FixedCandidate
import region_readout_reference as reference

def construct_regions(feature,mask_logits):
    h,w=feature.shape[-2:]
    m=mask_logits.detach().float()
    if m.shape[-2:]!=(h,w):m=F.interpolate(m,(h,w),mode='bilinear',align_corners=False)
    m=m.sigmoid(); padded=F.pad(m,(1,1,1,1),mode='replicate')
    dilation=F.max_pool2d(padded,3,stride=1); erosion=-F.max_pool2d(-padded,3,stride=1)
    return torch.cat((erosion,(dilation-erosion).clamp_min(0),(1-dilation).clamp_min(0)),dim=1)

def permutation(image_id,h,w):
    uid=int.from_bytes(hashlib.sha256(image_id.encode('utf-8')).digest()[:8],'little',signed=False)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261018,uid]))).permutation(h*w)

def move_regions(regions,condition,image_ids):
    b,r,h,w=regions.shape
    if condition=='NATIVE':return regions
    if condition=='SHIFT_HALF':return torch.roll(regions,(h//2,w//2),dims=(-2,-1))
    if condition=='IDENTITY_CHECK':indices=np.tile(np.arange(h*w),(b,1))
    elif condition=='PERMUTE_FIXED':indices=np.stack([permutation(i,h,w) for i in image_ids])
    else:raise ValueError(condition)
    ix=torch.from_numpy(indices).to(regions.device).unsqueeze(1).expand(-1,r,-1)
    return regions.flatten(2).gather(2,ix).reshape(b,r,h,w)

def tokens_from_regions(feature,regions,min_mass_fraction=1e-4):
    h,w=feature.shape[-2:]; mass=regions.sum(dim=(-2,-1)); valid=mass>=min_mass_fraction*h*w
    values=feature.float()
    with torch.autocast(device_type=feature.device.type,enabled=False):
        tokens=torch.einsum('brhw,bchw->brc',regions.float(),values)
        tokens=tokens/mass.clamp_min(1e-6).unsqueeze(-1)
    tokens=torch.where(valid.unsqueeze(-1),tokens,values.mean(dim=(-2,-1)).unsqueeze(1))
    return tokens,mass/(h*w),valid

@contextmanager
def completed_region_override(feature,mask,regions):
    original=reference.region_tokens
    def adapted(actual_feature,actual_mask,min_mass_fraction=1e-4):
        assert actual_feature.data_ptr()==feature.data_ptr() and actual_mask.data_ptr()==mask.data_ptr()
        return tokens_from_regions(actual_feature,regions,min_mass_fraction)
    reference.region_tokens=adapted
    try:yield
    finally:reference.region_tokens=original

def cache_batch(model,images,metadata):
    fused,attention=model._fuse_features(list(model.backbone(images)))
    pool=model._pool(fused,attention);mask=model.mask_head(fused)
    embedding=model.metadata_encoder(metadata.to(device=pool.device,dtype=pool.dtype))
    gate=1+torch.sigmoid(model.metadata_gate(embedding))
    regions=construct_regions(fused,mask)
    return fused,pool,mask,embedding,gate,regions

def intervened_forward(model,cached,condition,image_ids):
    fused,pool,mask,embedding,gate,regions=cached
    moved=move_regions(regions,condition,image_ids)
    before=regions.sum(dim=(-2,-1));after=moved.sum(dim=(-2,-1))
    valid_before=before>=1e-4*regions.shape[-2]*regions.shape[-1]
    valid_after=after>=1e-4*regions.shape[-2]*regions.shape[-1]
    assert torch.allclose(before,after,rtol=1e-5,atol=0),'Regional mass changed'
    assert torch.equal(valid_before,valid_after),'Region validity changed'
    assert torch.isfinite(moved).all()
    with completed_region_override(fused,mask,moved):
        regional,diag=model.readout(fused,mask,pool,embedding)
    pooled=regional*gate+embedding
    out={'logits':model.head(pooled),'mel_logits':model.mel_head(pooled).squeeze(1),'readout_diagnostics':diag}
    out['readout_diagnostics']={**diag,'regional_mass_relative_error':
        (after-before).abs()/before.abs().clamp_min(1e-12)}
    return out

def state_fingerprint(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        value=tensor.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
