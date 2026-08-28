#!/usr/bin/env python
"""Params / FLOPs / latency -- the evidence for the lightweight claim."""
import json

from _common import base_parser, get_config

import torch

from ledd.efficiency import full_report
from ledd.engine.train import build_model

if __name__ == "__main__":
    ap = base_parser(__doc__)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="runs/efficiency.json")
    args = ap.parse_args()
    cfg = get_config(args)

    model = build_model(cfg)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state))
    print(json.dumps(full_report(model, args.out), indent=2))
