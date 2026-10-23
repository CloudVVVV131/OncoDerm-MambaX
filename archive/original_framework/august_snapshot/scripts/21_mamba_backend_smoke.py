from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lesionmamba.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    output = ensure_dir("results/bioengineering_direct_baselines") / "mamba_backend_smoke.json"
    if not torch.cuda.is_available() and not args.allow_cpu:
        write_json({"status": "failed", "reason": "CUDA is required for the B6-M experiment"}, output)
        raise RuntimeError("CUDA is required for the B6-M experiment")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from mamba_ssm import Mamba

        model = Mamba(d_model=32, d_state=16, d_conv=4, expand=2).to(device)
        sample = torch.randn(2, 49, 32, device=device, requires_grad=True)
        result = model(sample)
        result.float().square().mean().backward()
        finite = bool(torch.isfinite(result).all() and torch.isfinite(sample.grad).all())
        status = {
            "status": "passed" if finite else "failed",
            "mamba_ssm": version("mamba-ssm"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": device,
            "shape": list(result.shape),
            "forward_and_backward_finite": finite,
        }
        write_json(status, output)
        if not finite:
            raise RuntimeError("Mamba smoke test produced non-finite output or gradient")
        print(f"mamba_backend_smoke status=passed version={status['mamba_ssm']} device={device}")
    except Exception as exc:
        write_json({"status": "failed", "device": device, "error": str(exc)}, output)
        raise


if __name__ == "__main__":
    main()
