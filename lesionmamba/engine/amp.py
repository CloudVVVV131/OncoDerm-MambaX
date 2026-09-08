from __future__ import annotations

from contextlib import nullcontext


def autocast_context(device: str, enabled: bool = True):
    if not enabled:
        return nullcontext()
    try:
        import torch

        return torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", enabled=enabled)
    except Exception:
        return nullcontext()
