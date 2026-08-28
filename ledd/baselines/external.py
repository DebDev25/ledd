"""Adapters for baselines we do NOT reimplement: FIRE and DIRE.

Both ship public code; reimplementing them is wasted effort and a reviewer risk.
We wrap their released models so our evaluation harness can score them on OUR
splits with OUR degradations.

FIRE  -- Chu et al., CVPR 2025, https://github.com/Chuchad/FIRE (MIT).
         Trains end-to-end through a frozen LDM autoencoder: expect ~3-5x our
         model's training cost. Schedule it FIRST in the baseline phase.
DIRE  -- Wang et al., ICCV 2023. Per-image DDIM inversion makes full training
         infeasible on free tiers; run inference-only with the released checkpoint
         on the test sets, or cite reported numbers.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn


class ExternalRepoWrapper(nn.Module):
    """Generic wrapper: put the cloned repo on sys.path, build their model, adapt I/O."""

    def __init__(self, repo_path: str, builder, ckpt: Optional[str] = None,
                 device: str = "cuda"):
        super().__init__()
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(
                f"{repo_path} not found. Clone the baseline repo first (see baselines/README.md)."
            )
        sys.path.insert(0, os.path.abspath(repo_path))
        self.model = builder()
        if ckpt:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state.get("model", state), strict=False)
        self.model.to(device).eval()

    def forward(self, x: torch.Tensor, **kw) -> Dict[str, torch.Tensor]:
        out = self.model(x)
        logit = out["logit"] if isinstance(out, dict) else out
        return {"logit": logit.squeeze(-1) if logit.ndim > 1 else logit}


def build_fire(repo_path: str = "third_party/FIRE", ckpt: Optional[str] = None,
               device: str = "cuda") -> ExternalRepoWrapper:
    def builder():
        from networks.fire import FIRE          # noqa: F401  (path provided by repo)

        return FIRE()

    return ExternalRepoWrapper(repo_path, builder, ckpt, device)


def build_dire(repo_path: str = "third_party/DIRE", ckpt: Optional[str] = None,
               device: str = "cuda") -> ExternalRepoWrapper:
    def builder():
        import torchvision

        m = torchvision.models.resnet50()
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m

    return ExternalRepoWrapper(repo_path, builder, ckpt, device)
