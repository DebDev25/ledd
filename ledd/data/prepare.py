"""Build the preprocessed crop archive from raw GenImage.

Why crop and not resize
-----------------------
Resizing imprints the resampling kernel's own spectral signature. GenImage
generators output at different native resolutions (128-1024px), so a resize-based
pipeline lets the frequency stream learn "which resampling kernel" -- i.e. generator
identity -- instead of "real vs fake". That silently inflates cross-generator
numbers. We therefore CENTER-CROP to `size`, and only resize when an image is
smaller than the crop (unavoidable; logged and reported).

Why lossless PNG
----------------
Re-encoding to JPEG injects DCT block artifacts -- exactly the kind of spectral
structure Stream 2 hunts for. The archive is written as PNG; JPEG only ever appears
as a deliberate augmentation/degradation.

Both rules are applied IDENTICALLY to reals and fakes (Chai et al.'s
preprocessing-equalization warning).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import List, Optional

from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class PrepareStats:
    generator: str
    label: str
    n_written: int
    n_upscaled: int          # images smaller than crop -> had to be resized
    n_failed: int


def list_images(folder: str) -> List[str]:
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(root, f))
    return sorted(out)


def center_crop_or_resize(img: Image.Image, size: int) -> tuple[Image.Image, bool]:
    """Center-crop to size x size. Only resizes if the image is too small."""
    img = img.convert("RGB")
    w, h = img.size
    upscaled = False
    if min(w, h) < size:
        # Unavoidable: scale up the short side, then crop. Logged in stats.
        scale = size / min(w, h)
        img = img.resize((max(size, int(round(w * scale))), max(size, int(round(h * scale)))), Image.BICUBIC)
        upscaled = True
        w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size)), upscaled


def prepare_split(
    src_dir: str,
    dst_dir: str,
    generator: str,
    label: str,
    n: int,
    size: int = 224,
    seed: int = 42,
) -> PrepareStats:
    """Sample `n` images from src_dir and write cropped PNGs to dst_dir/generator/label."""
    files = list_images(src_dir)
    rng = random.Random(f"{seed}:{generator}:{label}")
    rng.shuffle(files)
    files = files[:n]

    out_dir = os.path.join(dst_dir, generator, label)
    os.makedirs(out_dir, exist_ok=True)

    n_up = n_fail = 0
    for i, src in enumerate(tqdm(files, desc=f"{generator}/{label}", leave=False)):
        dst = os.path.join(out_dir, f"{i:06d}.png")
        if os.path.exists(dst):        # resumable: sessions die
            continue
        try:
            with Image.open(src) as im:
                crop, up = center_crop_or_resize(im, size)
                crop.save(dst, format="PNG", optimize=True)
            n_up += int(up)
        except Exception:
            n_fail += 1
    written = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
    return PrepareStats(generator, label, written, n_up, n_fail)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Build the 224 crop archive from raw GenImage.")
    ap.add_argument("--src", required=True, help="raw GenImage root")
    ap.add_argument("--dst", required=True, help="output archive root")
    ap.add_argument("--generators", nargs="+", required=True)
    ap.add_argument("--n-per-class", type=int, default=10000)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--layout",
        default="{gen}/{split}/{label}",
        help="relative path pattern inside --src; {gen},{split},{label} are substituted. "
             "GenImage ships as <gen>/imagenet_ai_.../train/{ai,nature}.",
    )
    ap.add_argument("--split", default="train")
    ap.add_argument("--label-names", nargs=2, default=["nature", "ai"],
                    help="folder names for (real, fake)")
    args = ap.parse_args(argv)

    stats = []
    for gen in args.generators:
        for label, folder in zip(("real", "fake"), args.label_names):
            rel = args.layout.format(gen=gen, split=args.split, label=folder)
            src = os.path.join(args.src, rel)
            if not os.path.isdir(src):
                print(f"!! missing {src} -- skipping")
                continue
            stats.append(prepare_split(src, args.dst, gen, label,
                                       args.n_per_class, args.size, args.seed))

    os.makedirs(args.dst, exist_ok=True)
    meta = {
        "size": args.size,
        "n_per_class": args.n_per_class,
        "seed": args.seed,
        "crop_policy": "center_crop_no_resize_unless_too_small",
        "format": "png_lossless",
        "stats": [asdict(s) for s in stats],
    }
    with open(os.path.join(args.dst, "archive_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    total = sum(s.n_written for s in stats)
    up = sum(s.n_upscaled for s in stats)
    print(f"\nWrote {total} images. Upscaled (short side < {args.size}): {up} "
          f"({100*up/max(total,1):.1f}%) -- report this in the paper if non-trivial.")


if __name__ == "__main__":
    main()
