from __future__ import annotations

import torch
from torch import nn

from lesionmamba.models.modules.dpmss import (
    BasicGlobalSelectiveScan,
    DPMSSBlock,
    GlobalCrossSelectiveScan,
    LocalLightweightSelectiveScan,
)
from lesionmamba.models.modules.gffn import ConvGFFN
from lesionmamba.models.modules.lmgf import LMGF
from lesionmamba.models.modules.mple import MPLE
from lesionmamba.models.modules.stem import LayerNorm2d


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x.div(keep) * (keep + torch.rand(shape, dtype=x.dtype, device=x.device)).floor()


class LesionMambaBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        use_mple: bool | str = True,
        scan_mode: str = "dual_path",
        use_lmgf: bool | str = True,
        window_size: int = 7,
        drop_path: float = 0.0,
        **scan_kwargs,
    ) -> None:
        super().__init__()
        self.use_mple = bool(use_mple) and use_mple != "false"
        self.use_lmgf = bool(use_lmgf) and use_lmgf not in {False, "false", "optional_off"}
        self.norm1 = LayerNorm2d(channels)
        use_dilation = use_mple not in {"simplified", False}
        self.mple = MPLE(channels, use_dilation=use_dilation) if self.use_mple else nn.Identity()
        if scan_mode == "dual_path":
            self.scan = DPMSSBlock(channels, window_size=window_size, **scan_kwargs)
        elif scan_mode in {"global_cross", "global_only"}:
            self.scan = GlobalCrossSelectiveScan(channels, **scan_kwargs)
        elif scan_mode == "local_lightweight":
            self.scan = LocalLightweightSelectiveScan(channels, **scan_kwargs)
        else:
            self.scan = BasicGlobalSelectiveScan(channels, **scan_kwargs)
        self.lmgf = LMGF(channels) if self.use_lmgf else None
        self.drop1 = DropPath(drop_path)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = ConvGFFN(channels)
        self.drop2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = x
        x_norm = self.norm1(x)
        f_loc = self.mple(x_norm) if self.use_mple else x_norm
        f_ssm = self.scan(f_loc)
        if self.lmgf is not None:
            x, attn = self.lmgf(f_loc, f_ssm, residual)
        else:
            x = residual + self.drop1(f_ssm)
            attn = None
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x, attn


class LesionMambaStage(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        stage_cfg: dict,
        drop_path_rates: list[float],
        **scan_kwargs,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                LesionMambaBlock(
                    channels=channels,
                    use_mple=stage_cfg.get("use_mple", True),
                    scan_mode=stage_cfg.get("scan_mode", "dual_path"),
                    use_lmgf=stage_cfg.get("use_lmgf", True),
                    window_size=int(stage_cfg.get("window_size", 7)),
                    drop_path=drop_path_rates[i],
                    **scan_kwargs,
                )
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        last_attn = None
        for block in self.blocks:
            x, attn = block(x)
            if attn is not None:
                last_attn = attn
        return x, last_attn
