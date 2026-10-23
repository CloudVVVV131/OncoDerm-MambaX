"""Frozen candidate forward plus clinically recomputed heads for batch-local caching."""
import sys
import torch
from p3_common import SOURCE
sys.path.insert(0, str(SOURCE / 'model_code'))
sys.path.insert(0, str(SOURCE))
from lesionmamba.models.oncoderm_mambax import OncoDermMambaX
from region_readout_reference import RegionReadout, SpatialCrossAttentionReadout


class FixedCandidate(OncoDermMambaX):
    def __init__(self, cfg, variant, seed):
        assert cfg['model']['pretrained'] is False
        super().__init__(cfg)
        self.variant = variant
        assert self.use_mask_aux and self.use_metadata and self.metadata_dim == 22
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(seed + 1000)
            if variant == 'XATTN':
                self.readout = SpatialCrossAttentionReadout(channels=768, hidden=256, heads=4, residual_limit=.5)
            else:
                self.readout = RegionReadout(channels=768, hidden=128,
                    condition_on_metadata=variant == 'R_CLIN', residual_limit=.5)

    def forward(self, x, metadata=None):
        features = self.backbone(x)
        fused, attention = self._fuse_features(list(features))
        original_pool = self._pool(fused, attention)
        mask_logits = self.mask_head(fused)
        if metadata is None or metadata.numel() == 0:
            metadata = torch.zeros((x.shape[0], self.metadata_dim), device=x.device, dtype=original_pool.dtype)
        metadata = metadata.to(device=x.device, dtype=original_pool.dtype)
        h = self.metadata_encoder(metadata)
        if self.variant == 'XATTN':
            pooled, diagnostics = self.readout(fused, original_pool, h)
        else:
            pooled, diagnostics = self.readout(fused, mask_logits, original_pool, h)
        pooled = pooled * (1.0 + torch.sigmoid(self.metadata_gate(h))) + h
        return {'logits': self.head(pooled), 'mel_logits': self.mel_head(pooled).squeeze(1),
                'attention': attention, 'mask_logits': mask_logits, 'features': fused,
                'readout_diagnostics': diagnostics}


def cache_image(model, x):
    fused, attention = model._fuse_features(list(model.backbone(x)))
    pool = model._pool(fused, attention)
    mask = model.mask_head(fused) if model.use_mask_aux else None
    return fused, attention, pool, mask


def cached_forward(model, cached, metadata):
    fused, attention, pool, mask = cached
    h = model.metadata_encoder(metadata.to(device=pool.device, dtype=pool.dtype))
    diagnostics = {}
    if isinstance(model, FixedCandidate):
        if model.variant == 'XATTN':
            pool, diagnostics = model.readout(fused, pool, h)
        else:
            pool, diagnostics = model.readout(fused, mask, pool, h)
    pool = pool * (1.0 + torch.sigmoid(model.metadata_gate(h))) + h
    return {'logits': model.head(pool), 'mel_logits': model.mel_head(pool).squeeze(1),
            'readout_diagnostics': diagnostics}
