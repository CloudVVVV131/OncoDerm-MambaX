"""Frozen P1 v1 integration, with training_seed + 1000 readout initialization."""
from pathlib import Path
import sys
import hashlib
import torch

OUT = Path(__file__).resolve().parent
PKG = OUT / 'FIM_RegionClinical_Dev_Start'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PKG / 'Reference_Only/model_code'))
sys.path.insert(0, str(PKG))
from lesionmamba.models.oncoderm_mambax import OncoDermMambaX
from region_readout_reference import RegionReadout, SpatialCrossAttentionReadout


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class P2Model(OncoDermMambaX):
    def __init__(self, cfg, variant, initial):
        assert variant in ['R_IMG', 'R_CLIN', 'XATTN']
        assert cfg['model']['pretrained'] is False
        super().__init__(cfg)
        assert self.use_mask_aux and self.use_metadata and self.metadata_dim == 22
        self.load_state_dict(initial, strict=True)
        own = self.state_dict()
        assert set(own) == set(initial)
        assert all(torch.equal(own[k], initial[k]) for k in own)
        self.base_key_mapping = {k: k for k in own}
        self.variant = variant
        # New CPU parameters use an isolated seed; preserve CPU and all CUDA RNGs.
        before_cpu = torch.get_rng_state().clone()
        before_cuda = torch.cuda.get_rng_state_all()
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(cfg['training_seed'] + 1000)
            if variant == 'XATTN':
                self.readout = SpatialCrossAttentionReadout(channels=768, hidden=256, heads=4, residual_limit=.5)
            else:
                self.readout = RegionReadout(channels=768, hidden=128,
                    condition_on_metadata=variant == 'R_CLIN', residual_limit=.5)
        self.initial_readout_hashes = {k: tensor_sha(v) for k,v in self.readout.state_dict().items()}
        assert torch.equal(before_cpu, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(before_cuda, torch.cuda.get_rng_state_all()))
        self.readout_rng_isolated = True
        assert all(torch.equal(self.state_dict()[k], initial[k]) for k in initial)

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
