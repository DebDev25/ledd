"""NPR -- Neighbouring Pixel Relationships (Tan et al., CVPR 2024).

Used two ways in this project:
  1. as an optional extra input channel for Stream 1 (ablation);
  2. as a standalone baseline (see ledd/baselines/npr_baseline.py).

Mechanism: CNN generators up-sample then convolve, which imprints a local
statistical dependency among neighbouring pixels. Subtracting a reference pixel
inside each l x l grid exposes it while cancelling image content.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def npr_map(x: torch.Tensor, grid: int = 2, reference: str = "first") -> torch.Tensor:
    """(B,C,H,W) -> (B,C,H,W) local pixel-difference map.

    Each grid cell has its reference pixel subtracted from every pixel in the cell.
    Default grid=2 matches the ubiquitous 2x up-sampling scale.
    """
    b, c, h, w = x.shape
    hh, ww = h // grid * grid, w // grid * grid
    x = x[..., :hh, :ww]
    cells = x.view(b, c, hh // grid, grid, ww // grid, grid)
    if reference == "first":
        ref = cells[:, :, :, :1, :, :1]
    elif reference == "mean":
        ref = cells.mean(dim=(3, 5), keepdim=True)
    else:
        raise ValueError(reference)
    return (cells - ref).view(b, c, hh, ww)


class NPRChannel(nn.Module):
    """Concatenates an NPR map to the RGB input and re-projects to 3 channels so a
    pretrained 3-channel backbone can still be used without surgery."""

    def __init__(self, grid: int = 2, reference: str = "first"):
        super().__init__()
        self.grid, self.reference = grid, reference
        self.proj = nn.Conv2d(6, 3, kernel_size=1)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():   # start as identity on RGB, NPR contribution learned
            w = torch.zeros(3, 6, 1, 1)
            w[0, 0] = w[1, 1] = w[2, 2] = 1.0
            self.proj.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = npr_map(x, self.grid, self.reference)
        if n.shape[-2:] != x.shape[-2:]:
            n = nn.functional.pad(n, (0, x.shape[-1] - n.shape[-1], 0, x.shape[-2] - n.shape[-2]))
        return self.proj(torch.cat([x, n], dim=1))
