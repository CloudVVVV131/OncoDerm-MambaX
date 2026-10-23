from __future__ import annotations

from typing import Any


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(config: dict[str, Any], split: str = "train"):
    try:
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("torchvision is required for image transforms") from exc

    data_cfg = config.get("data", {})
    input_size = int(data_cfg.get("input_size", 224))
    resize_short = int(data_cfg.get("resize_short_side", 256))
    tcfg = data_cfg.get("transforms", {})
    if split == "train":
        aug = [
            transforms.RandomResizedCrop(
                input_size, scale=tuple(tcfg.get("random_resized_crop_scale", [0.80, 1.00]))
            ),
            transforms.RandomHorizontalFlip(float(tcfg.get("hflip_p", 0.5))),
            transforms.RandomVerticalFlip(float(tcfg.get("vflip_p", 0.5))),
            transforms.RandomRotation(float(tcfg.get("rotation_degrees", 30))),
            transforms.ColorJitter(
                brightness=float(tcfg.get("color_jitter", [0.10, 0.10, 0.08, 0.02])[0]),
                contrast=float(tcfg.get("color_jitter", [0.10, 0.10, 0.08, 0.02])[1]),
                saturation=float(tcfg.get("color_jitter", [0.10, 0.10, 0.08, 0.02])[2]),
                hue=float(tcfg.get("color_jitter", [0.10, 0.10, 0.08, 0.02])[3]),
            ),
        ]
        if float(tcfg.get("gaussian_blur_p", 0.05)) > 0:
            aug.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=float(tcfg["gaussian_blur_p"])))
        aug.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        if float(tcfg.get("random_erasing_p", 0.0)) > 0:
            aug.append(transforms.RandomErasing(p=float(tcfg["random_erasing_p"])))
        return transforms.Compose(aug)
    return transforms.Compose(
        [
            transforms.Resize(resize_short),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
