"""Realistic post-upload degradations.

Used two ways:
  1. as training augmentation (destroys generator-specific noise, per CNNDetection);
  2. as the robustness stress set at eval time.

Everything operates in PIXEL space and is applied BEFORE the FFT, so the frequency
stream sees degraded spectra -- if you degrade after the FFT the frequency stream
trains on clean spectra it will never see at test time.
"""
from __future__ import annotations

import io
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def resize_down_up(img: Image.Image, scale: float) -> Image.Image:
    """Downscale then restore size -- simulates messaging-app resampling."""
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def brightness_contrast(img: Image.Image, b: float, c: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) * (1.0 + c) + 0.5 + b
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


# ---------------------------------------------------------------- named presets
# Fixed, reproducible degradations for the robustness table. GenImage's own
# degraded task uses JPEG q=65/30 and blur sigma=3/5; we keep compatible points.
EVAL_DEGRADATIONS = {
    "none": lambda im: im,
    "jpeg_95": lambda im: jpeg_compress(im, 95),
    "jpeg_75": lambda im: jpeg_compress(im, 75),
    "jpeg_65": lambda im: jpeg_compress(im, 65),
    "jpeg_50": lambda im: jpeg_compress(im, 50),
    "jpeg_30": lambda im: jpeg_compress(im, 30),
    "blur_1": lambda im: gaussian_blur(im, 1.0),
    "blur_2": lambda im: gaussian_blur(im, 2.0),
    "blur_3": lambda im: gaussian_blur(im, 3.0),
    "resize_0.5": lambda im: resize_down_up(im, 0.5),
    "resize_0.25": lambda im: resize_down_up(im, 0.25),
    "bright_0.2": lambda im: brightness_contrast(im, 0.2, 0.0),
    "contrast_0.3": lambda im: brightness_contrast(im, 0.0, 0.3),
}


def apply_named(img: Image.Image, name: str) -> Image.Image:
    if name not in EVAL_DEGRADATIONS:
        raise KeyError(f"Unknown degradation '{name}'. Options: {sorted(EVAL_DEGRADATIONS)}")
    return EVAL_DEGRADATIONS[name](img)


class RandomDegradation:
    """Stochastic training augmentation (CNNDetection-style blur+JPEG, plus resize)."""

    def __init__(
        self,
        jpeg_prob: float = 0.5,
        jpeg_quality: Tuple[int, int] = (50, 95),
        blur_prob: float = 0.3,
        blur_sigma: Tuple[float, float] = (0.5, 2.0),
        resize_prob: float = 0.3,
        resize_scale: Tuple[float, float] = (0.5, 0.9),
        color_jitter: float = 0.1,
    ):
        self.jpeg_prob, self.jpeg_quality = jpeg_prob, jpeg_quality
        self.blur_prob, self.blur_sigma = blur_prob, blur_sigma
        self.resize_prob, self.resize_scale = resize_prob, resize_scale
        self.color_jitter = color_jitter

    def __call__(self, img: Image.Image) -> Image.Image:
        # Order matters: blur -> resize -> jitter -> JPEG last, mirroring a real
        # upload pipeline where compression is the final step.
        if random.random() < self.blur_prob:
            img = gaussian_blur(img, random.uniform(*self.blur_sigma))
        if random.random() < self.resize_prob:
            img = resize_down_up(img, random.uniform(*self.resize_scale))
        if self.color_jitter > 0 and random.random() < 0.5:
            j = self.color_jitter
            img = brightness_contrast(img, random.uniform(-j, j), random.uniform(-j, j))
        if random.random() < self.jpeg_prob:
            img = jpeg_compress(img, random.randint(*self.jpeg_quality))
        return img
