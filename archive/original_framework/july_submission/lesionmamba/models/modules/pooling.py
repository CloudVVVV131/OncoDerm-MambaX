from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LesionAwarePoolingHead(nn.Module):
    def __init__(self, channels: int, num_classes: int = 7, use_lesion_pooling: bool = True) -> None:
        super().__init__()
        self.use_lesion_pooling = use_lesion_pooling
        in_features = channels * 3 if use_lesion_pooling else channels
        self.norm = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor, attention: torch.Tensor | None = None) -> torch.Tensor:
        if not self.use_lesion_pooling:
            z = x.mean(dim=(2, 3))
            return self.fc(self.norm(z))
        z_avg = x.mean(dim=(2, 3))
        z_max = x.amax(dim=(2, 3))
        if attention is None:
            attention = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        if attention.shape[-2:] != x.shape[-2:]:
            attention = F.interpolate(attention, size=x.shape[-2:], mode="bilinear", align_corners=False)
        denom = attention.sum(dim=(2, 3)).clamp_min(1e-6)
        z_les = (x * attention).sum(dim=(2, 3)) / denom
        z = torch.cat([z_avg, z_max, z_les], dim=1)
        return self.fc(self.norm(z))
