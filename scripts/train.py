#!/usr/bin/env python
"""Train any stage: spatial | frequency | joint (stage comes from the config).

  python scripts/train.py --config configs/stage1_spatial.yaml
  python scripts/train.py --config configs/stage2_frequency.yaml
  python scripts/train.py --config configs/stage3_joint.yaml
  python scripts/train.py --config configs/ablations/fusion_concat.yaml
"""
import json

from _common import base_parser, get_config

from ledd.engine.train import train

if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    cfg = get_config(args)
    print(json.dumps(train(cfg), indent=2))
