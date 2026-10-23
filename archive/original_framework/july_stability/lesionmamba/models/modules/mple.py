from __future__ import annotations

import torch
from torch import nn


class MPLE(nn.Module):
    def __init__(self, channels: int, use_dilation: bool = True) -> None:
        super().__init__()
        self.use_dilation = use_dilation
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.dw_dilated = (
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False)
            if use_dilation
            else None
        )
        self.act = nn.GELU()
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        if self.dw_dilated is not None:
            y = y + self.dw_dilated(x)
        return self.pw(self.act(y))
