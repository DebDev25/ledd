#!/usr/bin/env python
"""Audit the RAW dataset for the real-vs-fake format bias before preprocessing.

GenImage's reals are ImageNet JPEGs; several fake subsets are PNG. A detector can
score highly by learning "is this JPEG?" instead of anything about generators
(Ricker et al., arXiv:2403.17608). Run this on the raw download; if the format or
size distributions differ sharply between real and fake, use
`prepare_data.py --equalize-jpeg 95`.
"""
from __future__ import annotations

import argparse
import collections
import os
import random

from PIL import Image


def audit(folder: str, n: int = 300):
    files = []
    for root, _, fs in os.walk(folder):
        for f in fs:
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                files.append(os.path.join(root, f))
    random.Random(0).shuffle(files)
    fmt = collections.Counter()
    sizes = collections.Counter()
    kb = []
    for p in files[:n]:
        try:
            with Image.open(p) as im:
                fmt[im.format] += 1
                sizes[im.size] += 1
            kb.append(os.path.getsize(p) / 1024)
        except Exception:
            pass
    return {"n": len(files), "formats": dict(fmt),
            "top_sizes": sizes.most_common(3),
            "mean_kb": round(sum(kb) / max(len(kb), 1), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", required=True, help="raw real/nature folder")
    ap.add_argument("--fake", required=True, help="raw fake/ai folder")
    ap.add_argument("-n", type=int, default=300)
    args = ap.parse_args()

    r, f = audit(args.real, args.n), audit(args.fake, args.n)
    print("REAL:", r)
    print("FAKE:", f)
    same_format = set(r["formats"]) == set(f["formats"])
    print()
    if not same_format:
        print("BIAS: real and fake use different file formats "
              f"({sorted(r['formats'])} vs {sorted(f['formats'])}).")
        print("  -> run prepare_data.py with --equalize-jpeg 95 (this is the default).")
    else:
        print("Formats match. Still keep --equalize-jpeg on unless you have a reason not to.")


if __name__ == "__main__":
    main()
