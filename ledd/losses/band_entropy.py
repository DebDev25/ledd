"""Band-entropy regulariser -- the concrete form of 'frequency regularisation'.

Penalises LOW entropy in the CLS->band attention distribution, i.e. discourages the
frequency stream from collapsing onto a single band. Motivation: FrePGAN/NPR show
frequency features overfit to source-specific patterns, and GenImage shows the
high-frequency evidence dies under JPEG. A stream that has spread its reliance
across bands degrades more gracefully.

Keep the weight SMALL (~0.05). Forcing uniform attention would destroy the very
band-attribution signal the stream is built around -- we want to discourage
collapse, not mandate uniformity.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def band_entropy_loss(band_attn: torch.Tensor, normalize: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """band_attn: (B, n_bands), rows sum to 1. Returns a loss in [0,1] if normalised."""
    p = band_attn.clamp(min=eps)
    p = p / p.sum(dim=-1, keepdim=True)
    entropy = -(p * p.log()).sum(dim=-1)
    if normalize:
        entropy = entropy / math.log(p.shape[-1])
        return (1.0 - entropy).mean()          # 0 = perfectly spread, 1 = collapsed
    return (-entropy).mean()


class BandEntropyCriterion(nn.Module):
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, band_attn: torch.Tensor) -> torch.Tensor:
        return band_entropy_loss(band_attn, self.normalize)
