"""Explanation maps for the LEDD detector.

Produces, per image:
  * spatial_map      (H,W)          -- where in the image the decision came from
  * band_attribution (n_bands,)     -- which frequency rings mattered
  * band_to_region   (n_bands,H,W)  -- WHERE each band's evidence lives
                                        (from Freq->Spatial cross-attention;
                                         no prior paper reports this)
  * balance          (2,)           -- spatial vs frequency reliance, token-normalised

All of these are claims, not evidence. faithfulness.py is what tests them.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..models.attention import set_attention_recording
from .chefer import (collect_attention_blocks, cross_attention_relevance,
                     self_attention_relevance)


def _upsample(map_2d: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(map_2d.unsqueeze(1), size=(size, size), mode="bilinear",
                         align_corners=False).squeeze(1)


def explain_batch(model, x: torch.Tensor, target: str = "logit",
                  image_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Run a forward+backward pass with recording on and build all maps."""
    image_size = image_size or x.shape[-1]
    was_training = model.training
    model.eval()
    set_attention_recording(model, store_attn=True, store_grad=True)

    x = x.clone().requires_grad_(True)
    out = model(x)
    logit = out["logit"]
    model.zero_grad(set_to_none=True)
    logit.sum().backward(retain_graph=True)

    res: Dict[str, torch.Tensor] = {
        "logit": logit.detach(),
        "band_attribution": out["band_attention"].detach() if out.get("band_attention") is not None else None,
        "balance": out["balance"].detach() if out.get("balance") is not None else None,
    }

    # ---- frequency-side self-attention relevance (within Stream 2)
    freq_blocks = [b.attn for b in model.frequency.blocks]
    R_freq = self_attention_relevance(freq_blocks)

    # ---- [FUSE] readout: relevance over [spatial tokens | band tokens]
    readout = model.fusion.readout if hasattr(model.fusion, "readout") else None
    gh, gw = out["grid"]
    if readout is not None and readout.attn is not None:
        R_read = cross_attention_relevance(readout)              # (B,1,Ns+Nf)
        ns = gh * gw
        spatial_rel = R_read[:, 0, :ns].reshape(-1, gh, gw)
        band_rel = R_read[:, 0, ns:]
        res["spatial_map"] = _upsample(spatial_rel, image_size).detach()
        res["band_relevance"] = band_rel.detach()

    # ---- band-to-region maps from the Freq->Spatial cross-attention
    if hasattr(model.fusion, "f2s") and model.fusion.f2s is not None:
        blk = model.fusion.f2s[-1].attn
        if blk.attn is not None:
            R_cross = cross_attention_relevance(
                blk, R_q_self=R_freq[:, 1:, 1:] if R_freq is not None else None
            )                                                     # (B, n_bands, Ns)
            b, nb, ns = R_cross.shape
            maps = R_cross.reshape(b * nb, gh, gw)
            res["band_to_region"] = _upsample(maps, image_size).reshape(b, nb, image_size, image_size).detach()

    set_attention_recording(model, store_attn=False, store_grad=False)
    if was_training:
        model.train()
    return res


def normalize_map(m: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample min-max to [0,1] for plotting and for deletion ranking."""
    flat = m.flatten(1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    return ((flat - lo) / (hi - lo + eps)).view_as(m)
