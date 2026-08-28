#!/usr/bin/env python
"""Generate a tiny synthetic archive so the whole pipeline can be exercised before
GenImage is downloaded.

"Real" images get fractal (1/f) noise texture, which is roughly how natural image
spectra behave. "Fake" images are built at half resolution and nearest-neighbour
upsampled, which imprints the same kind of up-sampling artifact real CNN/diffusion
decoders leave (this is the mechanism NPR, Tan et al. CVPR 2024, exploits).

So the task is genuinely learnable and the frequency stream has real signal --
but the numbers mean NOTHING scientifically. This exists only to prove the code
runs end to end.

    python scripts/make_dummy_data.py --dst data/dummy_224 --n 60
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image


def fractal_noise(size: int, rng: np.random.Generator, beta: float = 1.8) -> np.ndarray:
    """1/f^beta noise -- a crude stand-in for natural image spectra."""
    white = rng.normal(size=(size, size, 3))
    f = np.fft.fftshift(np.fft.fft2(white, axes=(0, 1)), axes=(0, 1))
    cy = cx = (size - 1) / 2
    y = (np.arange(size) - cy)[:, None]
    x = (np.arange(size) - cx)[None, :]
    r = np.sqrt(y**2 + x**2)
    r[int(cy), int(cx)] = 1.0
    f = f / (r**(beta / 2))[:, :, None]
    img = np.real(np.fft.ifft2(np.fft.ifftshift(f, axes=(0, 1)), axes=(0, 1)))
    img = (img - img.min()) / (np.ptp(img) + 1e-8)
    return img


def make_real(size: int, rng: np.random.Generator) -> Image.Image:
    img = fractal_noise(size, rng)
    return Image.fromarray((img * 255).astype(np.uint8))


def make_fake(size: int, rng: np.random.Generator, factor: int = 2) -> Image.Image:
    """Built small, then nearest-neighbour upsampled -> up-sampling artifact."""
    small = fractal_noise(size // factor, rng)
    up = np.repeat(np.repeat(small, factor, axis=0), factor, axis=1)
    up = up + rng.normal(scale=0.01, size=up.shape)      # mild decoder noise
    up = np.clip(up, 0, 1)
    return Image.fromarray((up * 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dst", default="data/dummy_224")
    ap.add_argument("--generators", nargs="+",
                    default=["gen_a", "gen_b", "gen_c", "gen_d", "gen_e", "gen_f",
                             "gen_val", "gen_test"])
    ap.add_argument("--n", type=int, default=60, help="images per class per generator")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for gi, gen in enumerate(args.generators):
        rng = np.random.default_rng(args.seed + gi * 1000)
        # each "generator" gets a slightly different up-sampling factor / noise,
        # mimicking generator-specific fingerprints
        factor = 2 if gi % 2 == 0 else 4
        for label in ("real", "fake"):
            d = os.path.join(args.dst, gen, label)
            os.makedirs(d, exist_ok=True)
            for i in range(args.n):
                img = make_real(args.size, rng) if label == "real" else make_fake(args.size, rng, factor)
                img.save(os.path.join(d, f"{i:05d}.png"))
        print(f"{gen}: {args.n} real + {args.n} fake (upsample x{factor})")

    total = len(args.generators) * args.n * 2
    print(f"\nWrote {total} images to {args.dst}")
    print("NOTE: synthetic data. Use it to verify the pipeline runs, never to report results.")


if __name__ == "__main__":
    main()
