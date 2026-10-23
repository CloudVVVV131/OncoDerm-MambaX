from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lesionmamba.utils.io import ensure_dir


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    torch.save(state, p)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
