from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from lesionmamba.utils.io import ensure_dir


def find_last_conv2d(model: nn.Module) -> tuple[str, nn.Conv2d] | None:
    """Return the last Conv2d module, a robust default target for Grad-CAM."""

    target: tuple[str, nn.Conv2d] | None = None
    excluded = ("mask_head", "attention_head", "attention_from_feature")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if any(token in name for token in excluded):
                continue
            target = (name, module)
    return target


def compute_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int | None = None,
    target_layer: nn.Module | None = None,
    metadata: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, str]:
    """Compute a single-image Grad-CAM map.

    Args:
        model: Model returning either a logits tensor or a dict with ``logits``.
        image: Tensor shaped ``[1, 3, H, W]`` on the same device as the model.
        target_class: Optional class index. If omitted, the predicted class is used.
        target_layer: Optional layer. If omitted, the last Conv2d is selected.

    Returns:
        Normalized CAM tensor shaped ``[1, 1, H, W]``, target class index, and
        target layer name.
    """

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Grad-CAM expects a single image tensor shaped [1, 3, H, W].")

    layer_name = "custom"
    if target_layer is None:
        found = find_last_conv2d(model)
        if found is None:
            raise RuntimeError("No Conv2d layer found for Grad-CAM target selection.")
        layer_name, target_layer = found

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"Grad-CAM target layer {layer_name} did not return a tensor output.")
        activations.append(output)
        if output.requires_grad:
            output.register_hook(lambda grad: gradients.append(grad))

    f_handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        out = model(image, metadata=metadata) if metadata is not None else model(image)
        logits = out["logits"] if isinstance(out, dict) else out
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = logits[:, int(target_class)].sum()
        score.backward()
    finally:
        f_handle.remove()

    if not activations or not gradients:
        raise RuntimeError(f"Grad-CAM hooks did not capture activations/gradients for layer {layer_name}.")

    act = activations[-1].detach()
    grad = gradients[-1].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    cam_min = cam.amin(dim=(2, 3), keepdim=True)
    cam_max = cam.amax(dim=(2, 3), keepdim=True)
    cam = (cam - cam_min) / (cam_max - cam_min).clamp_min(1e-6)
    return cam.detach().cpu(), int(target_class), layer_name


def save_gradcam_heatmap(cam: torch.Tensor, path: str | Path) -> None:
    """Save a normalized CAM tensor as a red-yellow heatmap PNG."""

    path = Path(path)
    ensure_dir(path.parent)
    arr = cam.detach().float().cpu()
    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        arr = arr[0]
    arr_np = arr.numpy()
    arr_np = (arr_np - arr_np.min()) / max(arr_np.max() - arr_np.min(), 1e-6)
    red = np.ones_like(arr_np)
    green = arr_np
    blue = 1.0 - arr_np
    rgb = np.stack([red, green, blue], axis=-1)
    Image.fromarray(np.uint8(rgb * 255), mode="RGB").save(path)
