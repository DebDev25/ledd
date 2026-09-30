#!/usr/bin/env python
"""Pick a decision threshold on the VALIDATION GENERATOR, then apply it everywhere.

Why this exists
---------------
The raw logits are badly calibrated: a model with 0.946 AUC scored F1 0.775 at the
default threshold, because the logit distribution shifts on an unseen generator.
AUC is threshold-free and therefore unaffected, but every accuracy and F1 figure in
the paper depends on this step.

The threshold is fitted on the validation generator ONLY. Fitting it on the test
generator would leak, and fitting it in-distribution reproduces the same shift.

    python scripts/calibrate_threshold.py --config configs/kaggle.yaml \
        --ckpt runs/stage3_joint/best.pth --set data.root=$DATA
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from _common import base_parser, get_config
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ledd.data.dataset import build_loader
from ledd.data.splits import load_splits
from ledd.engine.evaluate import collect_predictions
from ledd.engine.train import build_model

if __name__ == "__main__":
    ap = base_parser(__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--fit-on", default="val_generator",
                    help="split used to choose the threshold; never the test split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = get_config(args)

    device = "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") != "cpu" else "cpu"
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"])
    model.eval()

    splits_path = args.splits or os.path.join(cfg["train"]["ckpt_dir"], "splits.json")
    splits = load_splits(splits_path)
    bs = cfg["eval"]["batch_size"]
    nw = cfg["data"].get("num_workers", 4)

    def predict(split):
        p = collect_predictions(model, build_loader(splits[split], bs, train=False,
                                                    num_workers=nw), device, cfg.get("amp", True))
        return np.asarray(p["logits"]), np.asarray(p["labels"])

    lv, yv = predict(args.fit_on)
    grid = np.quantile(lv, np.linspace(0.01, 0.99, 199))
    f1s = [f1_score(yv, (lv >= t).astype(int), zero_division=0) for t in grid]
    T = float(grid[int(np.argmax(f1s))])

    report = {"threshold": T, "fit_on": args.fit_on,
              "default_f1_at_zero": float(f1_score(yv, (lv >= 0).astype(int), zero_division=0)),
              "splits": {}}
    print(f"threshold {T:+.4f}   (F1 on {args.fit_on}: {max(f1s):.4f}, "
          f"was {report['default_f1_at_zero']:.4f} at logit 0)\n")

    for split in ("val_indist", "val_generator", "test_generator"):
        if split not in splits or not splits[split]:
            continue
        l, y = predict(split)
        pred = (l >= T).astype(int)
        m = {"auc": float(roc_auc_score(y, l)),
             "acc": float(accuracy_score(y, pred)),
             "f1": float(f1_score(y, pred, zero_division=0)),
             "n": int(len(y))}
        report["splits"][split] = m
        print(f"{split:16s} AUC {m['auc']:.4f}   acc {m['acc']:.4f}   F1 {m['f1']:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"\nwritten to {args.out}")
