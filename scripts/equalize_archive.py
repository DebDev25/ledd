#!/usr/bin/env python
"""Give every image in an EXISTING archive the same JPEG history, in place.

Use this only if scripts/audit_jpeg_history.py reports a shortcut. It re-encodes
both classes at one quality and saves back as PNG, so reals and fakes share an
identical final compression step -- no need to re-download or rebuild from raw.

Kaggle inputs are read-only, so write to a new directory and re-upload, or run this
on the copy you still have locally.

    python scripts/equalize_archive.py --src /kaggle/input/<slug> \
        --dst /kaggle/working/genimage_224_eq --quality 95
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil

from PIL import Image
from tqdm import tqdm


def equalize(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--generators", nargs="+", default=None)
    args = ap.parse_args()

    gens = args.generators or [d for d in sorted(os.listdir(args.src))
                               if os.path.isdir(os.path.join(args.src, d))]
    n = 0
    for gen in gens:
        for label in ("real", "fake"):
            sd = os.path.join(args.src, gen, label)
            dd = os.path.join(args.dst, gen, label)
            if not os.path.isdir(sd):
                continue
            os.makedirs(dd, exist_ok=True)
            files = [f for f in os.listdir(sd) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            for f in tqdm(files, desc=f"{gen}/{label}", leave=False):
                out = os.path.join(dd, os.path.splitext(f)[0] + ".png")
                if os.path.exists(out):
                    continue
                with Image.open(os.path.join(sd, f)) as im:
                    equalize(im, args.quality).save(out, format="PNG", optimize=True)
                n += 1
        print(f"{gen}: done")

    meta_src = os.path.join(args.src, "archive_meta.json")
    meta = json.load(open(meta_src)) if os.path.exists(meta_src) else {}
    meta["equalize_jpeg"] = args.quality
    meta["equalized_in_place_from"] = args.src
    meta["note"] = ("Both classes re-encoded at one JPEG quality so they share an "
                    "identical final compression step (arXiv:2403.17608). Reals retain "
                    "their original ImageNet JPEG history underneath; re-run "
                    "audit_jpeg_history.py to confirm the shortcut is gone.")
    json.dump(meta, open(os.path.join(args.dst, "archive_meta.json"), "w"), indent=2)
    print(f"\nRe-encoded {n} images at quality {args.quality} -> {args.dst}")
    print("Now re-run: python scripts/audit_jpeg_history.py --archive " + args.dst)


if __name__ == "__main__":
    main()
