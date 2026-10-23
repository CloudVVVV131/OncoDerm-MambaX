from __future__ import annotations

from torch import nn

from lesionmamba.models.baselines import build_timm_model
from lesionmamba.models.lesionmamba_compact import build_lesionmamba_compact
from lesionmamba.models.oncoderm_mambax import build_oncoderm_mambax
from lesionmamba.models.vmamba_adapter import build_vmamba_adapter


def build_model(config: dict) -> nn.Module:
    model_cfg = config.get("model", {})
    source = model_cfg.get("source", "timm")
    name = model_cfg.get("name", "")
    if source == "oncoderm_mambax" or name.startswith("oncoderm_mambax"):
        return build_oncoderm_mambax(config)
    if source == "lesionmamba" or name.startswith("lesionmamba"):
        return build_lesionmamba_compact(config)
    if source == "vmamba" or name.startswith("vmamba"):
        return build_vmamba_adapter(model_cfg)
    if source == "medmamba":
        raise RuntimeError("MedMamba is optional and requires external code. Use configs only after installing it.")
    return build_timm_model(model_cfg)


def create_model(name: str, **kwargs) -> nn.Module:
    """Convenience constructor for smoke tests and small utilities.

    The formal training pipeline should continue to call ``build_model`` with a
    full merged YAML config. This helper keeps one-off checks from duplicating
    the source/name dispatch logic.
    """

    model_cfg = dict(kwargs)
    model_cfg.setdefault("name", name)
    model_cfg.setdefault("num_classes", 7)
    if "source" not in model_cfg:
        if name.startswith("lesionmamba"):
            model_cfg["source"] = "lesionmamba"
        elif name.startswith("oncoderm_mambax"):
            model_cfg["source"] = "oncoderm_mambax"
        elif name.startswith("vmamba"):
            model_cfg["source"] = "vmamba"
        else:
            model_cfg["source"] = "timm"
            model_cfg.setdefault("timm_name", name)
    return build_model({"model": model_cfg})


def model_metadata(model: nn.Module) -> dict:
    return getattr(model, "metadata", {})
