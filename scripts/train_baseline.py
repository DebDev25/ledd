#!/usr/bin/env python
"""Train a baseline on the SAME splits, preprocessing and degradations as LEDD.

The model is selected by `stage: baseline` plus `baseline.name` in the config, which
build_model() resolves directly — so evaluate.py and calibrate_threshold.py work on
baseline checkpoints with the same config, without any patching.
"""
import json
import sys

from _common import base_parser, get_config

from ledd.engine.train import train

if __name__ == "__main__":
    ap = base_parser(__doc__)
    ap.add_argument("--baseline", default=None, help="overrides baseline.name in the config")
    args = ap.parse_args()
    cfg = get_config(args)

    cfg.setdefault("baseline", {})
    if args.baseline:
        cfg["baseline"]["name"] = args.baseline
    cfg["stage"] = "baseline"
    if not cfg["baseline"].get("name"):
        sys.exit("No baseline specified (--baseline, or baseline.name in the config)")

    print(json.dumps(train(cfg), indent=2))
