from __future__ import annotations

import warnings

import torch
from torch import nn


class SelectiveScan1D(nn.Module):
    """Unified 1D selective scan wrapper.

    Real VMamba/mamba kernels are environment-specific. This module records the
    requested backend and provides a deterministic torch fallback for smoke tests
    and development. Formal runs should set strict_ssm=True when a real backend
    is required.
    """

    def __init__(
        self,
        channels: int,
        backend: str = "auto",
        strict_ssm: bool = False,
        allow_torch_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.requested_backend = backend
        self.strict_ssm = strict_ssm
        self.allow_torch_fallback = allow_torch_fallback
        self.effective_backend = "torch_fallback"
        self.in_proj = nn.Linear(channels, channels * 2)
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.out_proj = nn.Linear(channels, channels)
        self._resolve_backend()

    def _resolve_backend(self) -> None:
        if self.requested_backend in {"vmamba", "auto"}:
            # Hook point for VMamba selective scan. We intentionally do not import
            # a guessed symbol here because public forks expose different names.
            pass
        if self.requested_backend in {"mamba_ssm", "auto"}:
            try:
                import mamba_ssm  # noqa: F401

                self.effective_backend = "mamba_ssm_available_torch_wrapper"
                return
            except Exception:
                pass
        if self.strict_ssm and not self.allow_torch_fallback:
            raise RuntimeError("No real selective scan backend is available and torch fallback is disabled.")
        if self.requested_backend not in {"torch_fallback", "auto"}:
            warnings.warn(
                f"Requested SSM backend {self.requested_backend!r} is unavailable; using torch_fallback. "
                "Do not report this as a formal VMamba/selective-scan result.",
                RuntimeWarning,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        y = value.transpose(1, 2)
        y = self.dw_conv(y).transpose(1, 2)
        y = y * torch.sigmoid(gate)
        return self.out_proj(y)


def sequence_from_feature(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


def feature_from_sequence(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], h, w).contiguous()
