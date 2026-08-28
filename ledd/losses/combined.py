"""Total objective:  L = L_BCE + w_con * L_SupCon + w_ent * L_band-entropy

Single-stage multi-task training (not SupCon's original two-stage recipe): simpler,
and the ablation table needs the joint version anyway.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .band_entropy import BandEntropyCriterion
from .supcon import SupConCriterion


class CombinedLoss(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        lc = cfg["loss"]
        self.w_bce = lc.get("bce_weight", 1.0)
        self.w_con = lc.get("supcon_weight", 0.5)
        self.w_ent = lc.get("band_entropy_weight", 0.05)

        sc = lc.get("supcon", {})
        self.supcon = SupConCriterion(
            dim=cfg["model"].get("projection", {}).get("dim", 128),
            temperature=sc.get("temperature", 0.1),
            queue_size=sc.get("queue_size", 8192),
            warmup_steps=sc.get("warmup_steps", 500),
            ramp_steps=sc.get("ramp_steps", 1000),
        ) if self.w_con > 0 else None

        self.band_entropy = BandEntropyCriterion(
            lc.get("band_entropy", {}).get("normalize", True)
        ) if self.w_ent > 0 else None

    def forward(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        logit = outputs["logit"]
        bce = F.binary_cross_entropy_with_logits(logit, labels.float())
        total = self.w_bce * bce
        parts = {"bce": bce.detach()}

        if self.supcon is not None and outputs.get("proj") is not None:
            l_con = self.supcon(outputs["proj"], labels)
            total = total + self.w_con * l_con
            parts["supcon"] = l_con.detach()

        if self.band_entropy is not None and outputs.get("band_attention") is not None:
            l_ent = self.band_entropy(outputs["band_attention"])
            total = total + self.w_ent * l_ent
            parts["band_entropy"] = l_ent.detach()

        parts["total"] = total.detach()
        return {"loss": total, **parts}
