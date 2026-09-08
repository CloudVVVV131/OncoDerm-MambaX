from __future__ import annotations

from copy import deepcopy

from torch import nn

from lesionmamba.models.baselines import build_timm_model
from lesionmamba.models.oncoderm_mambax import build_oncoderm_mambax


def build_model(config: dict) -> nn.Module:
    model_cfg = config.get("model", {})
    source = model_cfg.get("source", "timm")
    name = model_cfg.get("name", "")
    if source == "dermamamba_architecture_faithful" or name.startswith(
        "dermamamba_architecture_faithful"
    ):
        raise ValueError("The external DermaMamba implementation is not distributed in this repository.")
    if source == "oncoderm_mambax" or name.startswith("oncoderm_mambax"):
        return build_oncoderm_mambax(config)
    if source != "timm":
        raise ValueError(f"Unsupported public model source: {source}")
    return build_timm_model(model_cfg)


def build_model_for_checkpoint(config: dict) -> nn.Module:
    """Build an architecture with pretrained initialization disabled for checkpoint loading."""

    restore_config = deepcopy(config)
    restore_config.setdefault("model", {})["pretrained"] = False
    return build_model(restore_config)


def create_model(name: str, **kwargs) -> nn.Module:
    """Convenience constructor for smoke tests and small utilities.

    Training uses ``build_model`` with a full merged YAML config.
    This helper dispatches a model name and keyword arguments through that builder.
    """

    model_cfg = dict(kwargs)
    model_cfg.setdefault("name", name)
    model_cfg.setdefault("num_classes", 7)
    if "source" not in model_cfg:
        if name.startswith("oncoderm_mambax"):
            model_cfg["source"] = "oncoderm_mambax"
        else:
            model_cfg["source"] = "timm"
            model_cfg.setdefault("timm_name", name)
    return build_model({"model": model_cfg})


def model_metadata(model: nn.Module) -> dict:
    return getattr(model, "metadata", {})
