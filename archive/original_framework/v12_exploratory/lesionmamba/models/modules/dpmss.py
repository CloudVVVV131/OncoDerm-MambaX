from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lesionmamba.models.modules.selective_scan import SelectiveScan1D, feature_from_sequence, sequence_from_feature


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b, c, h // window_size, window_size, w // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, window_size * window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int, b: int) -> torch.Tensor:
    c = windows.shape[-1]
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, c, h, w)


class LocalWindowSelectiveScan(nn.Module):
    def __init__(self, channels: int, window_size: int = 7, **scan_kwargs) -> None:
        super().__init__()
        self.window_size = window_size
        self.scan = SelectiveScan1D(channels, **scan_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        x_pad = F.pad(x, (0, pad_w, 0, pad_h)) if pad_h or pad_w else x
        hp, wp = x_pad.shape[-2:]
        windows = window_partition(x_pad, ws)
        scanned = self.scan(windows)
        out = window_reverse(scanned, ws, hp, wp, b)
        return out[:, :, :h, :w].contiguous()


class LocalLightweightSelectiveScan(nn.Module):
    """Stage-1 local texture encoder used when DP-MSS is intentionally disabled."""

    def __init__(self, channels: int, **_scan_kwargs) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local(x)


class GlobalCrossSelectiveScan(nn.Module):
    def __init__(self, channels: int, **scan_kwargs) -> None:
        super().__init__()
        self.scans = nn.ModuleList([SelectiveScan1D(channels, **scan_kwargs) for _ in range(4)])
        self.proj = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        lr = feature_from_sequence(self.scans[0](sequence_from_feature(x)), h, w)
        rl_in = sequence_from_feature(torch.flip(x, dims=[3]))
        rl = torch.flip(feature_from_sequence(self.scans[1](rl_in), h, w), dims=[3])
        tb_in = x.permute(0, 1, 3, 2).contiguous()
        tb = feature_from_sequence(self.scans[2](sequence_from_feature(tb_in)), w, h).permute(0, 1, 3, 2)
        bt_in = torch.flip(tb_in, dims=[3])
        bt = torch.flip(feature_from_sequence(self.scans[3](sequence_from_feature(bt_in)), w, h), dims=[3]).permute(0, 1, 3, 2)
        return self.proj(torch.cat([lr, rl, tb, bt], dim=1))


class BasicGlobalSelectiveScan(nn.Module):
    def __init__(self, channels: int, **scan_kwargs) -> None:
        super().__init__()
        self.scan = SelectiveScan1D(channels, **scan_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return feature_from_sequence(self.scan(sequence_from_feature(x)), h, w)


class DPMSSBlock(nn.Module):
    def __init__(self, channels: int, window_size: int = 7, **scan_kwargs) -> None:
        super().__init__()
        self.local = LocalWindowSelectiveScan(channels, window_size, **scan_kwargs)
        self.global_cross = GlobalCrossSelectiveScan(channels, **scan_kwargs)
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_lw = self.local(x)
        f_gc = self.global_cross(x)
        alpha = torch.sigmoid(self.gate(torch.cat([f_lw, f_gc], dim=1)))
        return alpha * f_lw + (1.0 - alpha) * f_gc
