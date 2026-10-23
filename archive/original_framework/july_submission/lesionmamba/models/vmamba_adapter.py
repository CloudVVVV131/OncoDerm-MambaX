from __future__ import annotations

import warnings

from lesionmamba.models.baselines import DictOutputWrapper, build_timm_model


def build_vmamba_adapter(model_cfg: dict) -> DictOutputWrapper:
    """Adapter hook for VMamba.

    Public VMamba repositories vary in package names and constructors. This
    adapter records a clear fallback instead of pretending that a Swin/timm model
    is an actual VMamba result.
    """

    try:
        import vmamba  # type: ignore  # noqa: F401

        raise RuntimeError("VMamba package found but no stable constructor is configured in this adapter.")
    except Exception as exc:
        warnings.warn(f"VMamba backend unavailable or unconfigured: {exc}. Using explicit fallback model.", RuntimeWarning)
        fallback_cfg = {
            "name": model_cfg.get("name", "vmamba_tiny"),
            "timm_name": model_cfg.get("fallback_timm_name", "swin_tiny_patch4_window7_224"),
            "pretrained": bool(model_cfg.get("pretrained", True)),
            "num_classes": int(model_cfg.get("num_classes", 7)),
        }
        wrapper = build_timm_model(fallback_cfg)
        wrapper.metadata.update(
            {
                "source": "vmamba_adapter_fallback",
                "requested_source": "vmamba",
                "fallback_warning": str(exc),
                "not_formal_vmamba": True,
            }
        )
        return wrapper
