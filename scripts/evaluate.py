#!/usr/bin/env python
"""Run the full evaluation protocol table on a trained checkpoint.

  python scripts/evaluate.py --config configs/stage3_joint.yaml \
      --ckpt runs/stage3_joint/best.pth --ood-root data/ood --out runs/stage3_joint/protocol.json
"""
import json

from _common import base_parser, get_config

import torch

from ledd.engine import load_ood_items, run_protocol
from ledd.engine.train import build_model

if __name__ == "__main__":
    ap = base_parser(__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", default=None, help="defaults to <ckpt_dir>/splits.json")
    ap.add_argument("--ood-root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = get_config(args)

    device = cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state))
    model.eval()

    import os

    splits = args.splits or os.path.join(cfg["train"]["ckpt_dir"], "splits.json")
    ood = load_ood_items(args.ood_root) if args.ood_root else None
    res = run_protocol(model, cfg, splits, ood_items=ood, out_path=args.out, device=device)
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items() if kk != "per_generator"})
                      for k, v in res.items()}, indent=2))
