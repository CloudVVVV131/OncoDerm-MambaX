from __future__ import annotations

import torch
from torch import nn

from lesionmamba.models.modules.downsample import OverlapPatchMerging
from lesionmamba.models.modules.pooling import LesionAwarePoolingHead
from lesionmamba.models.modules.selective_scan import SelectiveScan1D
from lesionmamba.models.modules.stage import LesionMambaStage


DEFAULT_STAGE_SETTINGS = {
    "stage1": {"use_mple": True, "scan_mode": "local_lightweight", "use_dpmss": False, "use_lmgf": False},
    "stage2": {"use_mple": True, "scan_mode": "dual_path", "use_dpmss": True, "use_lmgf": True, "window_size": 7},
    "stage3": {"use_mple": True, "scan_mode": "dual_path", "use_dpmss": True, "use_lmgf": True, "window_size": 7},
    "stage4": {"use_mple": "simplified", "scan_mode": "global_cross", "use_dpmss": "global_only", "use_lmgf": "optional"},
}
from lesionmamba.models.modules.stem import OverlapPatchStem


class LesionMambaCompact(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config.get("model", config)
        self.config = model_cfg
        self.num_classes = int(model_cfg.get("num_classes", 7))
        channels = model_cfg.get("channels", [48, 96, 192, 384])
        depths = model_cfg.get("depths", [1, 2, 4, 1])
        drop_path_rate = float(model_cfg.get("drop_path_rate", 0.1))
        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        dp_idx = 0
        scan_kwargs = {
            "backend": model_cfg.get("ssm_backend", "auto"),
            "strict_ssm": bool(model_cfg.get("strict_ssm", False)),
            "allow_torch_fallback": bool(model_cfg.get("allow_torch_fallback", True)),
        }
        self.stem = OverlapPatchStem(3, channels[0])
        stages_cfg = {k: dict(v) for k, v in DEFAULT_STAGE_SETTINGS.items()}
        for key, value in model_cfg.get("stages", {}).items():
            stages_cfg[key] = {**stages_cfg.get(key, {}), **value}
        self.stage1 = LesionMambaStage(
            channels[0], depths[0], stages_cfg.get("stage1", {}), dpr[dp_idx : dp_idx + depths[0]], **scan_kwargs
        )
        dp_idx += depths[0]
        self.down1 = OverlapPatchMerging(channels[0], channels[1])
        self.stage2 = LesionMambaStage(
            channels[1], depths[1], stages_cfg.get("stage2", {}), dpr[dp_idx : dp_idx + depths[1]], **scan_kwargs
        )
        dp_idx += depths[1]
        self.down2 = OverlapPatchMerging(channels[1], channels[2])
        self.stage3 = LesionMambaStage(
            channels[2], depths[2], stages_cfg.get("stage3", {}), dpr[dp_idx : dp_idx + depths[2]], **scan_kwargs
        )
        dp_idx += depths[2]
        self.down3 = OverlapPatchMerging(channels[2], channels[3])
        self.stage4 = LesionMambaStage(
            channels[3], depths[3], stages_cfg.get("stage4", {}), dpr[dp_idx : dp_idx + depths[3]], **scan_kwargs
        )
        self.head = LesionAwarePoolingHead(
            channels[3], self.num_classes, use_lesion_pooling=bool(model_cfg.get("use_lesion_pooling", True))
        )
        self.metadata = {
            "model_name": model_cfg.get("name", "lesionmamba_compact"),
            "implementation_variant": model_cfg.get("implementation_variant", "full_dpmss"),
            "requested_ssm_backend": scan_kwargs["backend"],
            "pretraining_status": "from_scratch"
            if not model_cfg.get("pretrained_checkpoint")
            else "loaded_or_requested_checkpoint",
            "pretrained_checkpoint": model_cfg.get("pretrained_checkpoint"),
            "channels": channels,
            "depths": depths,
            "stage_settings": stages_cfg,
            "use_lesion_pooling": bool(model_cfg.get("use_lesion_pooling", True)),
        }
        self.metadata["effective_ssm_backends"] = self._collect_effective_ssm_backends()

    def _collect_effective_ssm_backends(self) -> dict[str, int]:
        backends: dict[str, int] = {}
        for module in self.modules():
            if isinstance(module, SelectiveScan1D):
                backends[module.effective_backend] = backends.get(module.effective_backend, 0) + 1
        return backends

    def forward(self, x: torch.Tensor, metadata: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        x = self.stem(x)
        x, attn1 = self.stage1(x)
        x = self.down1(x)
        x, attn2 = self.stage2(x)
        x = self.down2(x)
        x, attn3 = self.stage3(x)
        x = self.down3(x)
        x, attn4 = self.stage4(x)
        attention = attn4 if attn4 is not None else attn3 if attn3 is not None else attn2 if attn2 is not None else attn1
        logits = self.head(x, attention)
        return {"logits": logits, "attention": attention, "features": x}


def build_lesionmamba_compact(config: dict) -> LesionMambaCompact:
    return LesionMambaCompact(config)
