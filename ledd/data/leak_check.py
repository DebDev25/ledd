"""Preprocessing leak check -- RUN THIS BEFORE ANY GPU TRAINING.

The failure mode: each generator's images pass through a slightly different
resize/encode history. If that history is recoverable from the spectrum, the
frequency stream can score well by identifying the *pipeline* rather than the
*generator artifacts*, and every cross-generator number is inflated.

Test: take REAL images only, push them through each generator subset's
preprocessing path, and try to predict which path they came from using a simple
classifier on radial spectrum features. Real images are identical in content, so
above-chance accuracy means the pipeline itself leaks.

Interpretation:
  accuracy ~ 1/n_paths  -> clean.
  accuracy >> chance    -> fix the pipeline (crop instead of resize, equalize
                           encoders) before training anything.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


def radial_profile(img: Image.Image, n_bins: int = 32) -> np.ndarray:
    """Log-magnitude spectrum collapsed into equal-area radial bins."""
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.mgrid[0:h, 0:w]
    r = np.sqrt(((y - cy) / cy) ** 2 + ((x - cx) / cx) ** 2)
    r = np.clip(r / np.sqrt(2.0), 0, 1 - 1e-6)
    # equal-area bins: area ~ r^2, so bin on r^2
    idx = (r**2 * n_bins).astype(int)
    prof = np.zeros(n_bins, dtype=np.float32)
    for b in range(n_bins):
        m = idx == b
        prof[b] = mag[m].mean() if m.any() else 0.0
    return prof


def run_leak_check(
    paths_by_path_id: Dict[str, Sequence[str]],
    n_per_path: int = 300,
    n_bins: int = 32,
    seed: int = 42,
) -> Dict[str, float]:
    """paths_by_path_id: {'sd_v14_pipeline': [real image paths processed that way], ...}"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X, y, names = [], [], sorted(paths_by_path_id)
    for label, name in enumerate(names):
        files = list(paths_by_path_id[name])[:n_per_path]
        for p in files:
            with Image.open(p) as im:
                X.append(radial_profile(im, n_bins))
            y.append(label)
    X, y = np.stack(X), np.array(y)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, multi_class="auto"))
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    chance = 1.0 / len(names)
    acc = float(scores.mean())
    return {
        "paths": names,
        "n_paths": len(names),
        "accuracy": acc,
        "accuracy_std": float(scores.std()),
        "chance": chance,
        "leak_ratio": acc / chance,
        "verdict": "CLEAN" if acc < chance + 0.10 else "LEAK -- fix preprocessing before training",
    }


def main(argv: List[str] | None = None) -> None:
    import os

    ap = argparse.ArgumentParser(description="Detect preprocessing leakage across generator paths.")
    ap.add_argument("--archive", required=True, help="crop archive root")
    ap.add_argument("--generators", nargs="+", required=True)
    ap.add_argument("--n-per-path", type=int, default=300)
    ap.add_argument("--out", default="runs/leak_check.json")
    args = ap.parse_args(argv)

    groups = {}
    for g in args.generators:
        d = os.path.join(args.archive, g, "real")   # REAL images only -- content is comparable
        if os.path.isdir(d):
            groups[g] = [os.path.join(d, f) for f in sorted(os.listdir(d))[: args.n_per_path]]
    res = run_leak_check(groups, n_per_path=args.n_per_path)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    if res["verdict"].startswith("LEAK"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
