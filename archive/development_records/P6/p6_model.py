"""P6-v2 fixed four-slot readout: same tensors, region/global content contrast."""
from pathlib import Path
import sys
import hashlib
import torch
from torch import nn

OUT = Path(__file__).resolve().parent
PKG = OUT / 'FIM_RegionClinical_Dev_Start'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PKG / 'Reference_Only/model_code'))
sys.path.insert(0, str(PKG))
from lesionmamba.models.oncoderm_mambax import OncoDermMambaX
from region_readout_reference import region_tokens


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode('utf-8'))
        h.update(str(tensor.dtype).encode('ascii'))
        h.update(str(tuple(tensor.shape)).encode('ascii'))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class TokenPreservingReadout(nn.Module):
    def __init__(self, mode):
        super().__init__()
        assert mode in ['region', 'global']
        self.mode = mode
        self.slots = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128, bias=True), nn.GELU())
            for _ in range(4)
        ])
        self.output = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 768, bias=False))
        self.eta = nn.Parameter(torch.zeros(()))

    def forward(self, feature, mask_logits, original_pool):
        tokens, area, valid = region_tokens(feature, mask_logits, min_mass_fraction=1e-4)
        if self.mode == 'global':
            tokens = feature.float().mean(dim=(-2, -1)).unsqueeze(1).expand(-1, 3, -1)
        inputs = [original_pool, *tokens.unbind(1)]
        projected = [slot(value) for slot, value in zip(self.slots, inputs)]
        h = self.output(torch.cat(projected, dim=1))
        coefficient = .5 * self.eta.tanh()
        residual = coefficient * (h - original_pool)
        output = original_pool + residual
        with torch.no_grad():
            pool_norm = original_pool.float().norm(dim=1)
            residual_norm = residual.float().norm(dim=1)
            diag = {
                'region_area': area.detach(), 'region_valid': valid.detach(),
                'slot_token_norm': torch.stack([v.detach().float().norm(dim=1) for v in inputs], dim=1),
                'slot_projection_norm': torch.stack([v.detach().float().norm(dim=1) for v in projected], dim=1),
                'residual_coefficient': coefficient.detach(),
                'original_pool_norm': pool_norm, 'residual_vector_norm': residual_norm,
                'residual_to_pool_norm_ratio': residual_norm / pool_norm.clamp_min(1e-12),
            }
        return output, diag


class P6Model(OncoDermMambaX):
    def __init__(self, cfg, mode, initial):
        assert cfg['model']['pretrained'] is False
        assert cfg['training_seed'] == 42
        super().__init__(cfg)
        assert self.use_mask_aux and self.use_metadata and self.metadata_dim == 22
        self.load_state_dict(initial, strict=True)
        assert all(torch.equal(self.state_dict()[k], v) for k, v in initial.items())
        cpu_before = torch.get_rng_state().clone()
        cuda_before = [v.clone() for v in torch.cuda.get_rng_state_all()]
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(6042)
            self.readout = TokenPreservingReadout(mode)
        self.readout_rng_isolated = torch.equal(cpu_before, torch.get_rng_state()) and all(
            torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
        assert self.readout_rng_isolated
        self.initial_readout_hashes = {k: tensor_sha(v) for k, v in self.readout.state_dict().items()}
        assert all(torch.equal(self.state_dict()[k], v) for k, v in initial.items())

    def forward(self, x, metadata=None):
        features = self.backbone(x)
        fused, attention = self._fuse_features(list(features))
        original_pool = self._pool(fused, attention)
        mask_logits = self.mask_head(fused)
        pooled, diagnostics = self.readout(fused, mask_logits, original_pool)
        if metadata is None or metadata.numel() == 0:
            metadata = torch.zeros((x.shape[0], self.metadata_dim), device=x.device, dtype=original_pool.dtype)
        metadata = metadata.to(device=x.device, dtype=original_pool.dtype)
        clinical = self.metadata_encoder(metadata)
        pooled = pooled * (1.0 + torch.sigmoid(self.metadata_gate(clinical))) + clinical
        return {'logits': self.head(pooled), 'mel_logits': self.mel_head(pooled).squeeze(1),
                'attention': attention, 'mask_logits': mask_logits, 'features': fused,
                'readout_diagnostics': diagnostics}
