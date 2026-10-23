from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance


def gaussian_blur(img: Image.Image, kernel: int) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=max(kernel // 2, 1)))


def brightness_shift(img: Image.Image, shift: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(1.0 + shift)


def hair_occlusion(img: Image.Image, lines: int = 5, seed: int = 42) -> Image.Image:
    rng = random.Random(seed)
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for _ in range(lines):
        x1, y1 = rng.randint(0, w), rng.randint(0, h)
        x2, y2 = rng.randint(0, w), rng.randint(0, h)
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=max(1, min(w, h) // 150))
    return out
