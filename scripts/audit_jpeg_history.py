#!/usr/bin/env python
"""Measure whether real and fake images differ in JPEG compression HISTORY.

Why a format check is not enough
--------------------------------
PNG is lossless, so converting an ImageNet JPEG to PNG preserves its 8x8 DCT block
artifacts exactly. An archive can be PNG/PNG throughout and still carry the full
real-vs-fake compression bias described in Ricker et al. (arXiv:2403.17608). That
bias is especially dangerous here: JPEG's 8x8 quantisation leaves a strong periodic
signature precisely where the band-token frequency stream looks.

This script computes a blockiness statistic per image and reports how well that
SINGLE NUMBER separates real from fake. Interpretation:

    |AUC - 0.5| < 0.05   clean -- compression history is not a shortcut
    0.05 - 0.15          mild  -- note it in the paper, JPEG augmentation may cover it
    > 0.15               strong shortcut -- equalise before training

    python scripts/audit_jpeg_history.py --archive /kaggle/input/<slug>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
from PIL import Image


def blockiness(img: Image.Image) -> float:
    """Ratio of pixel discontinuity ACROSS 8x8 block borders to WITHIN blocks.

    JPEG quantises each 8x8 block independently, which perturbs this ratio; images
    that never went through JPEG sit near a different value.
    """
    a = np.asarray(img.convert("L"), dtype=np.float64)
    out = []
    for arr in (a, a.T):
        d = np.abs(np.diff(arr, axis=1))
        cols = np.arange(d.shape[1])
        across = d[:, cols % 8 == 7].mean()
        within = d[:, cols % 8 != 7].mean()
        out.append(across / (within + 1e-8))
    return float(np.mean(out))


def sample_scores(folder: str, n: int, seed: int = 0):
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    picks = random.Random(seed).sample(files, min(n, len(files)))
    scores = []
    for f in picks:
        try:
            with Image.open(os.path.join(folder, f)) as im:
                scores.append(blockiness(im))
        except Exception:
            pass
    return scores


def auc(pos, neg) -> float:
    """Probability a random 'real' scores above a random 'fake' (rank-based, no sklearn)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--generators", nargs="+",
                    default=["sd_v14", "sd_v15", "wukong", "vqdm", "biggan",
                             "glide", "adm", "midjourney"])
    ap.add_argument("-n", type=int, default=200, help="images per class per generator")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    all_real, all_fake, rows = [], [], []
    for gen in args.generators:
        r = sample_scores(os.path.join(args.archive, gen, "real"), args.n)
        f = sample_scores(os.path.join(args.archive, gen, "fake"), args.n, seed=1)
        if not r or not f:
            print(f"  skipping {gen} (missing images)")
            continue
        a = auc(r, f)
        rows.append((gen, np.mean(r), np.mean(f), a))
        all_real += r
        all_fake += f

    if not rows:
        print("No data found.")
        return 1

    print(f"{'generator':<14}{'real':>9}{'fake':>9}{'AUC':>8}   verdict")
    for gen, mr, mf, a in rows:
        dev = abs(a - 0.5)
        v = "clean" if dev < 0.05 else ("mild" if dev < 0.15 else "SHORTCUT")
        print(f"{gen:<14}{mr:>9.4f}{mf:>9.4f}{a:>8.3f}   {v}")

    overall = auc(all_real, all_fake)
    dev = abs(overall - 0.5)
    print(f"\nPooled AUC from blockiness alone: {overall:.3f}  (|AUC-0.5| = {dev:.3f})")

    if dev < 0.05:
        verdict = ("CLEAN -- compression history is not a usable shortcut. "
                   "Do NOT regenerate; just record the setting in archive_meta.json.")
        code = 0
    elif dev < 0.15:
        verdict = ("MILD -- report it in the paper. Your JPEG augmentation (p=0.5, q50-95) "
                   "partially covers this, but consider equalising.")
        code = 0
    else:
        verdict = ("SHORTCUT PRESENT -- a single scalar separates the classes. Run "
                   "scripts/equalize_archive.py, then re-run this audit.")
        code = 1
    print(verdict)

    if args.out:
        json.dump({"pooled_auc": overall, "deviation": dev,
                   "per_generator": [{"generator": g, "real": mr, "fake": mf, "auc": a}
                                     for g, mr, mf, a in rows],
                   "verdict": verdict}, open(args.out, "w"), indent=2)
    return code


if __name__ == "__main__":
    sys.exit(main())
