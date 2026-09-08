from __future__ import annotations

import warnings

import torch
from torch import nn


class DictOutputWrapper(nn.Module):
    def __init__(self, model: nn.Module, name: str, metadata: dict | None = None) -> None:
        super().__init__()
        self.model = model
        self.name = name
        self.metadata = metadata or {}

    def forward(self, x: torch.Tensor, metadata: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        logits = self.model(x)
        if isinstance(logits, dict):
            return logits
        return {"logits": logits, "attention": None, "features": None}


def _build_torchvision(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    import torchvision.models as tvm

    weights = "DEFAULT" if pretrained else None
    if name == "resnet50":
        model = tvm.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"No torchvision fallback implemented for {name}")


def build_timm_model(model_cfg: dict) -> DictOutputWrapper:
    name = model_cfg.get("name", "baseline")
    timm_name = model_cfg.get("timm_name", name)
    pretrained = bool(model_cfg.get("pretrained", True))
    allow_random_init_fallback = bool(model_cfg.get("allow_random_init_fallback", True))
    num_classes = int(model_cfg.get("num_classes", 7))
    try:
        import timm

        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
        metadata = {"source": "timm", "timm_name": timm_name, "pretrained": pretrained}
    except Exception as exc:
        if timm_name == "resnet50":
            warnings.warn(f"timm unavailable, using torchvision fallback for resnet50: {exc}", RuntimeWarning)
            model = _build_torchvision("resnet50", num_classes, pretrained)
            metadata = {"source": "torchvision_fallback", "timm_name": timm_name, "pretrained": pretrained}
        else:
            if pretrained and not allow_random_init_fallback:
                raise RuntimeError(
                    f"Pretrained weights are required for {timm_name}, but loading failed. "
                    "Configuration: allow_random_init_fallback=False. "
                    f"Original error: {exc}"
                ) from exc
            try:
                import timm

                warnings.warn(
                    f"timm pretrained weights unavailable for {timm_name}; using random initialization: {exc}",
                    RuntimeWarning,
                )
                model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
                metadata = {
                    "source": "timm_random_init_fallback",
                    "timm_name": timm_name,
                    "pretrained": False,
                    "pretrained_fallback_reason": str(exc),
                }
            except Exception as fallback_exc:
                raise RuntimeError(f"Failed to build timm model {timm_name}: {fallback_exc}") from fallback_exc
    return DictOutputWrapper(model, name, metadata)
