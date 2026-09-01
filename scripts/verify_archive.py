#!/usr/bin/env python
"""Verify a preprocessed crop archive before training on it.

Checks the things that silently ruin an experiment: missing generators, class
imbalance, wrong image size, mixed formats, duplicate files, and whether the
JPEG-equalisation setting was recorded.

    python scripts/verify_archive.py --archive /kaggle/input/<slug>
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys

from PIL import Image

EXPECTED = ["sd_v14", "sd_v15", "wukong", "vqdm", "biggan", "midjourney", "adm", "glide"]


def scan(archive: str, gens, size: int, sample: int):
    problems, rows = [], []
    for gen in gens:
        row = {"generator": gen}
        for label in ("real", "fake"):
            d = os.path.join(archive, gen, label)
            if not os.path.isdir(d):
                problems.append(f"MISSING: {gen}/{label}")
                row[label] = 0
                continue
            files = [f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            row[label] = len(files)

            picks = random.Random(0).sample(files, min(sample, len(files)))
            fmts, sizes = collections.Counter(), collections.Counter()
            for f in picks:
                try:
                    with Image.open(os.path.join(d, f)) as im:
                        fmts[im.format] += 1
                        sizes[im.size] += 1
                except Exception as e:
                    problems.append(f"UNREADABLE: {gen}/{label}/{f} ({e})")
            row[f"{label}_fmt"] = "/".join(sorted(fmts))
            bad = [s for s in sizes if s != (size, size)]
            if bad:
                problems.append(f"WRONG SIZE in {gen}/{label}: {bad[:3]} (expected {size}x{size})")

        if row.get("real") and row.get("fake"):
            ratio = row["real"] / row["fake"]
            if not 0.8 <= ratio <= 1.25:
                problems.append(f"IMBALANCE in {gen}: {row['real']} real vs {row['fake']} fake")
        rows.append(row)
    return rows, problems


def duplicate_check(archive: str, gens, per_class: int = 200):
    """Duplicate images across real/fake or across generators would leak."""
    seen, dupes = {}, []
    for gen in gens:
        for label in ("real", "fake"):
            d = os.path.join(archive, gen, label)
            if not os.path.isdir(d):
                continue
            files = sorted(os.listdir(d))[:per_class]
            for f in files:
                p = os.path.join(d, f)
                try:
                    with open(p, "rb") as fh:
                        h = hashlib.md5(fh.read()).hexdigest()
                except Exception:
                    continue
                if h in seen and seen[h] != f"{gen}/{label}":
                    dupes.append(f"{seen[h]} == {gen}/{label}/{f}")
                seen[h] = f"{gen}/{label}"
    return dupes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--generators", nargs="+", default=EXPECTED)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--sample", type=int, default=50)
    args = ap.parse_args()

    if not os.path.isdir(args.archive):
        print(f"No such directory: {args.archive}")
        return 1

    rows, problems = scan(args.archive, args.generators, args.size, args.sample)
    warnings: list[str] = []

    w = max(len(r["generator"]) for r in rows) + 2
    print(f"{'generator':<{w}}{'real':>8}{'fake':>8}  formats")
    total = 0
    for r in rows:
        total += r.get("real", 0) + r.get("fake", 0)
        print(f"{r['generator']:<{w}}{r.get('real',0):>8}{r.get('fake',0):>8}  "
              f"{r.get('real_fmt','-')}/{r.get('fake_fmt','-')}")
    print(f"\nTotal images: {total:,}")

    meta_path = os.path.join(args.archive, "archive_meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        print(f"archive_meta.json: size={meta.get('size')} "
              f"n_per_class={meta.get('n_per_class')} "
              f"equalize_jpeg={meta.get('equalize_jpeg')}")
        if not meta.get("equalize_jpeg"):
            # NOT a hard failure: a PNG/PNG archive may still be clean, and it may
            # not be. Format alone cannot tell -- PNG losslessly preserves JPEG
            # artifacts. Measure it rather than assume either way.
            warnings.append("equalize_jpeg is OFF. This is not automatically a fault, but "
                            "PNG/PNG does not prove the archive is clean -- PNG preserves "
                            "source JPEG artifacts exactly. Decide with:\n"
                            "      python scripts/audit_jpeg_history.py --archive <archive>")
    else:
        problems.append("archive_meta.json missing -- preprocessing settings not recorded, "
                        "which you need for the paper's reproducibility section")

    print("\nChecking for duplicate images...")
    dupes = duplicate_check(args.archive, args.generators)
    if dupes:
        problems.append(f"DUPLICATES across classes/generators: {len(dupes)} found, e.g. {dupes[:2]}")
    else:
        print("  none in the sampled subset")

    print()
    for w_ in warnings:
        print("  ~ " + w_)
    if problems:
        for p in problems:
            print("  ! " + p)
        print(f"\n{len(problems)} blocking issue(s). Fix before training.")
        return 1
    if warnings:
        print(f"\n{len(warnings)} warning(s), 0 blocking issues.")
    else:
        print("Archive looks good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
