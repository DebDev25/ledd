#!/usr/bin/env python
"""Train a baseline on the SAME splits/preprocessing/degradations as LEDD."""
import json
import os
import sys

from _common import base_parser, get_config

import torch

from ledd.baselines import build_baseline
from ledd.engine import train as train_mod

if __name__ == "__main__":
    ap = base_parser(__doc__)
    ap.add_argument("--baseline", default=None, help="overrides config baseline.name")
    args = ap.parse_args()
    cfg = get_config(args)
    name = args.baseline or cfg.get("baseline", {}).get("name")
    if not name:
        sys.exit("No baseline specified (--baseline or baseline.name in config)")

    kwargs = {k: v for k, v in cfg.get("baseline", {}).items() if k != "name"}
    # Reuse the stage trainer by swapping in the baseline constructor.
    train_mod.build_model = lambda c, _n=name, _k=kwargs: build_baseline(_n, **_k)
    print(json.dumps(train_mod.train(cfg), indent=2))
