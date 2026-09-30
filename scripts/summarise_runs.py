#!/usr/bin/env python
"""Collect every training run into one table.

Reports BOTH the best epoch and the mean over epochs. The mean is the honest
number: selecting the best epoch rewards whichever run happened to be noisiest,
which is exactly what made cross-attention look better than concatenation before
the curves were examined.

    python scripts/summarise_runs.py --runs /kaggle/working/runs
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--last-k", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.runs, "*", "metrics.csv"))):
        name = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path)
        if "val_gen_auc" not in df or df["val_gen_auc"].isna().all():
            continue
        v = df["val_gen_auc"].dropna()
        i = v.idxmax()
        rows.append({
            "run": name,
            "epochs": len(df),
            "best": round(float(v.max()), 4),
            "mean": round(float(v.mean()), 4),
            f"last{args.last_k}": round(float(v.tail(args.last_k).mean()), 4),
            "swing": round(float(v.max() - v.min()), 4),
            "indist_at_best": round(float(df.loc[i, "indist_auc"]), 4),
            "gap": round(float(df.loc[i, "indist_auc"] - v.max()), 4),
            "mins": round(float(df["secs"].sum()) / 60, 1),
        })

    if not rows:
        raise SystemExit(f"no runs with metrics found under {args.runs}")

    table = pd.DataFrame(rows).sort_values("mean", ascending=False)
    print(table.to_string(index=False))
    print("\n'mean' is the number to report; 'best' flatters noisy runs.")
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"written to {args.out}")
