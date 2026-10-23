from __future__ import annotations

import torch
from torch import nn


class LMGF(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.att_dw = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2)
        self.att_pw = nn.Conv2d(channels * 2, 1, kernel_size=1)
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)

    def forward(self, f_loc: torch.Tensor, f_mss: torch.Tensor, f_res: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = torch.sigmoid(self.att_pw(self.att_dw(torch.cat([f_loc, f_mss], dim=1))))
        f_guide = f_mss * (1.0 + m) + f_loc * m
        f_fuse = self.fuse(torch.cat([f_guide, f_mss, f_loc], dim=1))
        return f_res + f_fuse, m
