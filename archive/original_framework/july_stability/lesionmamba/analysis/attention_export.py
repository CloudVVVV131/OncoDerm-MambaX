from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lesionmamba.utils.io import ensure_dir


def save_attention_map(attention: torch.Tensor, path: str | Path, size: tuple[int, int] = (224, 224)) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    att = attention.detach().float()
    if att.ndim == 4:
        att = att[0]
    if att.ndim == 3:
        att = att.unsqueeze(0)
    att = F.interpolate(att, size=size, mode="bilinear", align_corners=False)[0, 0]
    arr = att.cpu().numpy()
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)
    img = Image.fromarray(np.uint8(arr * 255), mode="L")
    img.save(path)
