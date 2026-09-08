from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from lesionmamba.data.datasets import metadata_dim as infer_metadata_dim


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1, groups: int = 1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchSelectiveScan2D(nn.Module):
    """Axial depthwise-convolution token mixer for CSSA.

    Horizontal and vertical Conv1d outputs are fused with a pointwise Conv2d.
    The effective backend is recorded in the model metadata.
    """

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.h_scan = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels, bias=False)
        self.v_scan = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels, bias=False)
        self.mix = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_seq = x.permute(0, 2, 1, 3).reshape(b * h, c, w)
        h_out = self.h_scan(h_seq).reshape(b, h, c, w).permute(0, 2, 1, 3)
        v_seq = x.permute(0, 3, 1, 2).reshape(b * w, c, h)
        v_out = self.v_scan(v_seq).reshape(b, w, c, h).permute(0, 2, 3, 1)
        return self.mix(torch.cat([h_out, v_out], dim=1))


class OfficialMambaTokenMixer2D(nn.Module):
    """Flatten a single feature map and mix its tokens with state-spaces/mamba.

    The B6-M configuration selects this backend and requires mamba_ssm.Mamba.
    """

    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as exc:
            raise RuntimeError(
                "B6-M requires the official state-spaces/mamba package. Install "
                "mamba-ssm with the commands in docs/ENVIRONMENT.md."
            ) from exc
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)
        seq = seq + self.mamba(self.norm(seq))
        return seq.transpose(1, 2).reshape(b, c, h, w)


class CrossScaleSelectiveStateAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        adapter_dim: int,
        scan_kernel_size: int = 7,
        backend: str = "axial_conv1d",
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ) -> None:
        super().__init__()
        self.backend = str(backend)
        self.proj_in = ConvNormAct(in_channels, adapter_dim, 1)
        self.local = ConvNormAct(adapter_dim, adapter_dim, 3, groups=adapter_dim)
        if self.backend == "mamba_ssm":
            self.scan = OfficialMambaTokenMixer2D(
                adapter_dim,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
        elif self.backend == "axial_conv1d":
            self.scan = TorchSelectiveScan2D(adapter_dim, scan_kernel_size)
        else:
            raise ValueError(f"Unsupported CSSA backend: {self.backend}")
        self.attention_head = nn.Conv2d(adapter_dim, 1, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(adapter_dim * 2 + 1, adapter_dim, 1),
            nn.Sigmoid(),
        )
        self.proj_out = nn.Conv2d(adapter_dim, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.proj_in(x)
        local = self.local(u)
        global_ctx = self.scan(u)
        attn = torch.sigmoid(self.attention_head(global_ctx + local))
        gate = self.gate(torch.cat([local, global_ctx, attn], dim=1))
        fused = gate * global_ctx + (1.0 - gate) * local
        out = x + self.gamma * self.proj_out(fused)
        return out, attn


class MetadataEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class OncoDermOutputs:
    logits: torch.Tensor
    mel_logits: torch.Tensor | None
    attention: torch.Tensor | None
    mask_logits: torch.Tensor | None
    features: torch.Tensor


class OncoDermMambaX(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        metadata_cfg = data_cfg.get("metadata", {})
        self.name = model_cfg.get("name", "oncoderm_mambax")
        self.num_classes = int(model_cfg.get("num_classes", data_cfg.get("num_classes", 7)))
        self.use_cssa = bool(model_cfg.get("cssa", {}).get("enabled", True))
        self.use_mask_aux = bool(model_cfg.get("lesion_prior", {}).get("enabled", True))
        self.use_metadata = bool(model_cfg.get("metadata_fusion", {}).get("enabled", True)) and bool(
            metadata_cfg.get("enabled", False)
        )
        self.use_mel_head = bool(model_cfg.get("heads", {}).get("melanoma_endpoint", True))
        self.use_prev_fusion = bool(model_cfg.get("multi_scale_fusion", self.use_cssa))
        self.metadata_dim = int(metadata_cfg.get("metadata_dim") or infer_metadata_dim(metadata_cfg))

        try:
            import timm

            timm_name = model_cfg.get("timm_name", model_cfg.get("backbone", "convnext_tiny"))
            pretrained = bool(model_cfg.get("pretrained", True))
            out_indices = tuple(model_cfg.get("out_indices", [1, 2, 3]))
            self.backbone = timm.create_model(
                timm_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
            )
            channels = list(self.backbone.feature_info.channels())
            self.metadata = {
                "model_name": self.name,
                "source": "oncoderm_mambax",
                "backbone_source": "timm_features",
                "timm_name": timm_name,
                "pretrained": pretrained,
                "out_indices": list(out_indices),
            }
        except Exception as exc:
            raise RuntimeError(f"Failed to build OncoDerm-MambaX feature backbone: {exc}") from exc

        if len(channels) < 1:
            raise RuntimeError("Feature backbone returned no feature channels.")
        self.channels = channels
        last_ch = channels[-1]
        prev_ch = channels[-2] if len(channels) > 1 else channels[-1]
        adapter_dim = int(model_cfg.get("cssa", {}).get("adapter_dim", min(256, last_ch)))
        cssa_cfg = model_cfg.get("cssa", {})
        scan_kernel = int(cssa_cfg.get("scan_kernel_size", 7))
        cssa_backend = str(cssa_cfg.get("backend", "axial_conv1d"))
        adapter_kwargs = {
            "scan_kernel_size": scan_kernel,
            "backend": cssa_backend,
            "mamba_d_state": int(cssa_cfg.get("d_state", 16)),
            "mamba_d_conv": int(cssa_cfg.get("d_conv", 4)),
            "mamba_expand": int(cssa_cfg.get("expand", 2)),
        }
        self.cssa_last = (
            CrossScaleSelectiveStateAdapter(last_ch, adapter_dim, **adapter_kwargs) if self.use_cssa else None
        )
        self.cssa_prev = (
            CrossScaleSelectiveStateAdapter(prev_ch, max(adapter_dim // 2, 64), **adapter_kwargs)
            if self.use_cssa and bool(model_cfg.get("cssa", {}).get("stage3_enabled", True)) and len(channels) > 1
            else None
        )

        fuse_dim = int(model_cfg.get("fusion_dim", last_ch))
        self.prev_proj = nn.Conv2d(prev_ch, last_ch, 1) if len(channels) > 1 else nn.Identity()
        fuse_in = last_ch * 2 if self.use_prev_fusion and len(channels) > 1 else last_ch
        self.fuse = ConvNormAct(fuse_in, fuse_dim, 1)
        self.mask_head = nn.Sequential(
            ConvNormAct(fuse_dim, max(fuse_dim // 2, 64), 3),
            nn.Conv2d(max(fuse_dim // 2, 64), 1, 1),
        )
        self.attention_from_feature = nn.Conv2d(fuse_dim, 1, 1)

        metadata_hidden = int(model_cfg.get("metadata_fusion", {}).get("hidden_dim", 64))
        self.metadata_encoder = (
            MetadataEncoder(self.metadata_dim, metadata_hidden, fuse_dim, float(model_cfg.get("dropout", 0.1)))
            if self.use_metadata and self.metadata_dim > 0
            else None
        )
        self.metadata_gate = nn.Linear(fuse_dim, fuse_dim) if self.metadata_encoder is not None else None
        head_in = fuse_dim
        dropout = float(model_cfg.get("dropout", 0.1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, self.num_classes))
        self.mel_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, 1)) if self.use_mel_head else None
        try:
            mamba_version = version("mamba-ssm") if cssa_backend == "mamba_ssm" else None
        except PackageNotFoundError:
            mamba_version = None
        effective_backend = "state_spaces_mamba_ssm" if cssa_backend == "mamba_ssm" else "torch_fallback_conv1d_scan"
        self.metadata.update(
            {
                "implementation_variant": "pretrained_backbone_cssa_lpah_cmgf",
                "cssa_enabled": self.use_cssa,
                "lpah_enabled": self.use_mask_aux,
                "cmgf_enabled": self.use_metadata,
                "melanoma_endpoint_head": self.use_mel_head,
                "multi_scale_fusion": self.use_prev_fusion,
                "metadata_dim": self.metadata_dim,
                "effective_ssm_backend": effective_backend,
                "formal_mamba_block": cssa_backend == "mamba_ssm",
                "mamba_ssm_version": mamba_version,
                "mamba_scope": "F4_only" if cssa_backend == "mamba_ssm" and self.cssa_prev is None else "multi_stage",
                "inference_requires_mask": False,
            }
        )

    def _fuse_features(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        last = features[-1]
        prev = features[-2] if len(features) > 1 else None
        attns: list[torch.Tensor] = []
        if self.cssa_last is not None:
            last, attn_last = self.cssa_last(last)
            attns.append(attn_last)
        if prev is not None and self.cssa_prev is not None:
            prev, attn_prev = self.cssa_prev(prev)
            attns.append(attn_prev)
        if prev is not None and self.use_prev_fusion:
            last_up = F.interpolate(last, size=prev.shape[-2:], mode="bilinear", align_corners=False)
            fused = self.fuse(torch.cat([self.prev_proj(prev), last_up], dim=1))
        else:
            fused = self.fuse(last)
        if attns:
            resized = [F.interpolate(a, size=fused.shape[-2:], mode="bilinear", align_corners=False) for a in attns]
            attention = torch.stack(resized, dim=0).mean(dim=0)
        else:
            attention = torch.sigmoid(self.attention_from_feature(fused))
        return fused, attention

    def _pool(self, feature: torch.Tensor, attention: torch.Tensor | None) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        if attention is None:
            return avg
        weights = attention.clamp(0, 1)
        weighted = (feature * weights).sum(dim=(2, 3)) / weights.sum(dim=(2, 3)).clamp_min(1e-6)
        return 0.5 * (avg + weighted)

    def forward(self, x: torch.Tensor, metadata: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        features = self.backbone(x)
        fused, attention = self._fuse_features(list(features))
        pooled = self._pool(fused, attention)
        if self.metadata_encoder is not None:
            if metadata is None or metadata.numel() == 0:
                metadata = torch.zeros((x.shape[0], self.metadata_dim), device=x.device, dtype=pooled.dtype)
            metadata = metadata.to(device=x.device, dtype=pooled.dtype)
            meta = self.metadata_encoder(metadata)
            gate = torch.sigmoid(self.metadata_gate(meta))
            pooled = pooled * (1.0 + gate) + meta
        logits = self.head(pooled)
        mel_logits = self.mel_head(pooled).squeeze(1) if self.mel_head is not None else None
        mask_logits = self.mask_head(fused) if self.use_mask_aux else None
        return {
            "logits": logits,
            "mel_logits": mel_logits,
            "attention": attention,
            "mask_logits": mask_logits,
            "features": fused,
        }


def build_oncoderm_mambax(config: dict[str, Any]) -> OncoDermMambaX:
    return OncoDermMambaX(config)
