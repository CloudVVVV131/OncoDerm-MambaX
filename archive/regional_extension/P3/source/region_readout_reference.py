"""Candidate regional readouts for the isolated Frontiers research line.

This file defines new, UNTRAINED hypotheses, not a validated clinical method.
Integration with the user's trainer is deliberately left to the local Codex task.
Inputs are the original B6 fused feature map, original pre-CMGF pooled feature,
and the ORIGINAL encoder's clinical embedding (computed exactly once).
A predicted mask is supplied by the original mask head, not by a ground-truth mask.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def region_tokens(feature: Tensor, mask_logits: Tensor,
                  min_mass_fraction: float = 1e-4) -> tuple[Tensor, Tensor, Tensor]:
    """Return Bx3xC tokens, Bx3 area fractions and Bx3 valid flags.

    Regions = soft eroded interior, morphological boundary band, outer context.
    Detachment stops the classifier's DIRECT gradient through region weights;
    it does not freeze the backbone or isolate its shared updates from classification.
    Empty regions use the image's global mean as a finite numerical placeholder.
    """
    if feature.ndim != 4 or mask_logits.ndim != 4:
        raise ValueError("feature and predicted mask_logits must have four dimensions")
    b, c, h, w = feature.shape
    if mask_logits.shape[0] != b or mask_logits.shape[1] != 1:
        raise ValueError("mask_logits must have shape [batch, 1, height, width]")
    if min(h, w) < 1 or min_mass_fraction <= 0:
        raise ValueError("positive spatial dimensions and mass threshold required")
    m = mask_logits.detach().float()
    if m.shape[-2:] != (h, w):
        m = F.interpolate(m, (h, w), mode="bilinear", align_corners=False)
    m = m.sigmoid()
    padded = F.pad(m, (1, 1, 1, 1), mode="replicate")
    dilation = F.max_pool2d(padded, 3, stride=1)
    erosion = -F.max_pool2d(-padded, 3, stride=1)
    regions = torch.cat((erosion, (dilation - erosion).clamp_min(0),
                         (1 - dilation).clamp_min(0)), dim=1)
    mass = regions.sum(dim=(-2, -1))
    valid = mass >= min_mass_fraction * h * w
    values = feature.float()
    # Keep reductions FP32 even when called inside CUDA autocast.
    with torch.autocast(device_type=feature.device.type, enabled=False):
        tokens = torch.einsum("brhw,bchw->brc", regions.float(), values)
        tokens = tokens / mass.clamp_min(1e-6).unsqueeze(-1)
    global_mean = values.mean(dim=(-2, -1)).unsqueeze(1)
    tokens = torch.where(valid.unsqueeze(-1), tokens, global_mean)
    return tokens, mass / (h * w), valid


class RegionReadout(nn.Module):
    """R_IMG or R_CLIN; the sole contrast is clinical information in region scores.

    Existing CMGF must still be applied AFTER this readout in both variants.
    Return value is pre-CMGF. The scalar residual multiplier is bounded, but this
    is not a calibrated confidence estimate or a guarantee of robustness.
    """
    def __init__(self, channels: int = 768, hidden: int = 128,
                 condition_on_metadata: bool = True,
                 residual_limit: float = 0.5) -> None:
        super().__init__()
        if channels <= 0 or hidden <= 0 or residual_limit <= 0:
            raise ValueError("channels, hidden and residual_limit must be positive")
        self.channels = channels
        self.condition_on_metadata = condition_on_metadata
        self.residual_limit = residual_limit
        self.token_norm = nn.LayerNorm(channels)
        self.global_norm = nn.LayerNorm(channels)
        self.meta_norm = nn.LayerNorm(channels)
        self.token_projection = nn.Linear(channels, hidden, bias=False)
        self.global_projection = nn.Linear(channels, hidden, bias=False)
        self.meta_projection = nn.Linear(channels, hidden, bias=False)
        self.region_type = nn.Parameter(torch.zeros(3, hidden))
        self.score = nn.Linear(hidden, 1, bias=False)
        self.output_projection = nn.Linear(channels, channels, bias=False)
        nn.init.eye_(self.output_projection.weight)
        # Only this multiplier is zero: zeroing BOTH it and the output projection
        # would create an avoidable dead residual branch.
        self.eta = nn.Parameter(torch.zeros(()))

    def forward(self, feature: Tensor, mask_logits: Tensor, original_pool: Tensor,
                metadata_embedding: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        b = feature.shape[0]
        if original_pool.shape != (b, self.channels):
            raise ValueError("original_pool must be BxC, before original CMGF")
        if metadata_embedding.shape != original_pool.shape:
            raise ValueError("supply the original BxC metadata embedding, not raw fields")
        tokens, area, valid = region_tokens(feature, mask_logits)
        hidden = self.token_projection(self.token_norm(tokens))
        hidden = hidden + self.global_projection(self.global_norm(original_pool)).unsqueeze(1)
        if self.condition_on_metadata:
            hidden = hidden + self.meta_projection(self.meta_norm(metadata_embedding)).unsqueeze(1)
        hidden = hidden + self.region_type.unsqueeze(0)
        logits = self.score(hidden.tanh()).squeeze(-1).float()
        logits = logits.masked_fill(~valid, torch.finfo(torch.float32).min)
        alpha = logits.softmax(dim=1)
        z = (alpha.unsqueeze(-1) * tokens).sum(dim=1)
        delta = self.output_projection(z - original_pool.float())
        coefficient = self.residual_limit * self.eta.tanh()
        out = original_pool + coefficient * delta.to(original_pool.dtype)
        return out, {"region_weights": alpha.detach(), "region_area": area.detach(),
                     "region_valid": valid.detach(), "residual_coefficient": coefficient.detach()}


class SpatialCrossAttentionReadout(nn.Module):
    """Standard clinical-query/grid-token reference, not a full paper reproduction.

    Same original B6, mask supervision, residual insertion and downstream CMGF.
    It receives no predicted-region prior. One query attends to the native grid.
    """
    def __init__(self, channels: int = 768, hidden: int = 256, heads: int = 4,
                 residual_limit: float = 0.5) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.channels, self.residual_limit = channels, residual_limit
        self.feature_norm = nn.LayerNorm(channels)
        self.meta_norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, hidden)
        self.grid = nn.Linear(channels, hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.output_projection = nn.Linear(hidden, channels)
        self.eta = nn.Parameter(torch.zeros(()))

    def forward(self, feature: Tensor, original_pool: Tensor,
                metadata_embedding: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if feature.ndim != 4 or original_pool.shape != metadata_embedding.shape:
            raise ValueError("expected BCHW features and two matching BxC embeddings")
        grid = self.grid(self.feature_norm(feature.flatten(2).transpose(1, 2)))
        query = self.query(self.meta_norm(metadata_embedding)).unsqueeze(1)
        token, _ = self.attention(query, grid, grid, need_weights=False)
        delta = self.output_projection(token[:, 0]) - original_pool
        coefficient = self.residual_limit * self.eta.tanh()
        return original_pool + coefficient * delta, {"residual_coefficient": coefficient.detach()}
