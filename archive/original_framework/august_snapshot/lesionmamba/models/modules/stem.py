from __future__ import annotations

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class OverlapPatchStem(nn.Module):
    def __init__(self, in_chans: int = 3, out_chans: int = 48, act_layer: type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(out_chans),
            act_layer(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, groups=out_chans, bias=False),
            nn.Conv2d(out_chans, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
