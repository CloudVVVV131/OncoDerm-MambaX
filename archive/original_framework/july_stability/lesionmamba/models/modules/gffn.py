from __future__ import annotations

import torch
from torch import nn


class ConvGFFN(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.in_proj = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.out_proj = nn.Conv2d(hidden, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.in_proj(x).chunk(2, dim=1)
        return self.out_proj(self.act(self.dw(u)) * torch.sigmoid(v))
