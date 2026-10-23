from __future__ import annotations

from torch import nn

from lesionmamba.models.modules.stem import LayerNorm2d


class OverlapPatchMerging(nn.Module):
    def __init__(self, in_chans: int, out_chans: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.GELU(),
            LayerNorm2d(out_chans),
        )

    def forward(self, x):
        return self.down(x)
