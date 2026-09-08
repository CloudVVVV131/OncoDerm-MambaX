from __future__ import annotations

import platform
import subprocess
from typing import Any


def environment_summary() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = str(torch.__version__)
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:
        info["torch_error"] = str(exc)
    for package in ["timm", "numpy", "pandas", "sklearn", "torchvision", "mamba_ssm"]:
        try:
            mod = __import__(package)
            info[package] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[package] = "not_installed"
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2)
        if git.returncode == 0:
            info["git_commit"] = git.stdout.strip()
    except Exception:
        pass
    return info
