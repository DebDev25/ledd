#!/usr/bin/env python
"""Explainability + faithfulness evaluation.

Produces:
  * deletion/insertion AUC on spatial maps (vs a random-saliency control)
  * band deletion/insertion AUC (the novel frequency-band attribution)
  * validation of the spatial-vs-frequency balance metric against causal deletion
  * qualitative figures (spatial map, band attribution, band-to-region)
"""
import json
import os

from _common import base_parser, get_config

import torch

from ledd.data.dataset import build_loader
from ledd.data.splits import load_splits
from ledd.engine.train import build_model
from ledd.explain import (deletion_insertion_bands, deletion_insertion_pixels,
                          explain_batch, normalize_map, random_baseline_maps,
                          validate_balance_metric)

if __name__ == "__main__":
    ap = base_parser(__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test_generator")
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="runs/explain")
    ap.add_argument("--figures", type=int, default=8, help="how many qualitative examples")
    args = ap.parse_args()
    cfg = get_config(args)

    device = "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") != "cpu" else "cpu"
    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state))
    model.eval()

    splits = load_splits(os.path.join(cfg["train"]["ckpt_dir"], "splits.json"))
    loader = build_loader(splits[args.split], args.batch_size, train=False,
                          num_workers=cfg["data"].get("num_workers", 2))

    os.makedirs(args.out, exist_ok=True)
    agg = {"pixel": [], "pixel_random": [], "band": []}

    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        x = batch["image"].to(device)
        maps = explain_batch(model, x)

        if maps.get("spatial_map") is not None:
            sal = normalize_map(maps["spatial_map"])
            agg["pixel"].append(deletion_insertion_pixels(model, x, sal))
            agg["pixel_random"].append(
                deletion_insertion_pixels(model, x, random_baseline_maps(sal.shape, device))
            )
        if maps.get("band_attribution") is not None:
            agg["band"].append(deletion_insertion_bands(model, x, maps["band_attribution"]))

    def mean_of(key, field):
        vals = [d[field] for d in agg[key] if field in d]
        return float(sum(vals) / len(vals)) if vals else None

    summary = {
        "spatial": {
            "deletion_auc": mean_of("pixel", "deletion_auc"),
            "insertion_auc": mean_of("pixel", "insertion_auc"),
            "random_deletion_auc": mean_of("pixel_random", "deletion_auc"),
            "random_insertion_auc": mean_of("pixel_random", "insertion_auc"),
        },
        "frequency_bands": {
            "deletion_auc": mean_of("band", "band_deletion_auc"),
            "insertion_auc": mean_of("band", "band_insertion_auc"),
        },
        "balance_validation": validate_balance_metric(model, loader, device),
        "note": "deletion: LOWER is better; insertion: HIGHER is better. "
                "Maps must beat the random control or they are not explanations.",
    }
    with open(os.path.join(args.out, "faithfulness.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
