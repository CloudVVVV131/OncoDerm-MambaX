from __future__ import annotations

import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn


class LogitsOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out["logits"] if isinstance(out, dict) else out


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(checkpoint_path: str | Path | None) -> float:
    if checkpoint_path and Path(checkpoint_path).exists():
        return os.path.getsize(checkpoint_path) / (1024 * 1024)
    return float("nan")


def estimate_flops(model: torch.nn.Module, input_size: int = 224) -> float:
    try:
        from thop import profile

        dummy = torch.randn(1, 3, input_size, input_size, device=next(model.parameters()).device)
        flops, _ = profile(LogitsOnlyWrapper(model), inputs=(dummy,), verbose=False)
        return float(flops)
    except Exception:
        return float("nan")


@torch.no_grad()
def latency_throughput(
    model: torch.nn.Module,
    input_size: int = 224,
    batch_size_latency: int = 1,
    batch_size_throughput: int = 64,
    warmup_iters: int = 30,
    measure_iters: int = 100,
    amp: bool = True,
) -> dict[str, float]:
    device = next(model.parameters()).device
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def measure(batch_size: int) -> float:
        x = torch.randn(batch_size, 3, input_size, input_size, device=device)
        amp_enabled = amp and device.type in {"cuda", "cpu"}

        def autocast_ctx():
            return torch.autocast(device_type=device.type, enabled=amp_enabled) if amp_enabled else nullcontext()

        for _ in range(warmup_iters):
            with autocast_ctx():
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(measure_iters):
            with autocast_ctx():
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / max(measure_iters, 1)

    latency = measure(batch_size_latency) / batch_size_latency
    throughput_time = measure(batch_size_throughput)
    peak_mem = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0.0
    return {
        "latency_ms_b1": latency * 1000,
        "throughput_img_s": batch_size_throughput / max(throughput_time, 1e-9),
        "peak_mem_mb": peak_mem,
    }
