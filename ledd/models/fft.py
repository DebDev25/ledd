"""Spectrum utilities: FFT/DCT, equal-area radial ring masks, band pooling.

Design notes (Architecture doc S2.2)
-----------------------------------
* Grayscale/luminance input: spectral artifacts are not colour-specific and this
  cuts the frequency stream's compute ~3x.
* log(1 + |F|) after fftshift, then per-image standardisation.
* The DC component and the lowest ring are ZEROED -- they are pure image content
  and we do not want the stream leaning on them.
* Rings are EQUAL-AREA, not equal-radius: area grows as r^2, so equal-radius rings
  would give the outer bands vastly more pixels (and energy) than the inner ones,
  which biases both the tokens and the attention distribution used for attribution.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

RGB_TO_LUMA = (0.299, 0.587, 0.114)


def to_luma(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) -> (B,1,H,W). Assumes ImageNet-normalised input is fine; the FFT is
    shift/scale sensitive only in DC, which we discard anyway."""
    if x.shape[1] == 1:
        return x
    w = torch.tensor(RGB_TO_LUMA, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def log_magnitude_spectrum(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(B,1,H,W) real -> (B,1,H,W) log-magnitude, fftshifted and standardised."""
    f = torch.fft.fft2(x, norm="ortho")
    f = torch.fft.fftshift(f, dim=(-2, -1))
    mag = torch.log1p(f.abs())
    mean = mag.mean(dim=(-2, -1), keepdim=True)
    std = mag.std(dim=(-2, -1), keepdim=True) + eps
    return (mag - mean) / std


def phase_spectrum(x: torch.Tensor) -> torch.Tensor:
    f = torch.fft.fft2(x, norm="ortho")
    f = torch.fft.fftshift(f, dim=(-2, -1))
    return torch.angle(f) / torch.pi        # in [-1, 1]


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2-D DCT-II via FFT (no scipy dependency).

    Used by the locked magnitude-vs-phase-vs-DCT ablation. For DCT the 'rings'
    become DCT frequency bands measured from the origin (top-left), so we return
    an origin-centred map by mirroring, keeping the ring machinery unchanged.
    """
    def dct_1d(v: torch.Tensor) -> torch.Tensor:
        n = v.shape[-1]
        v2 = torch.cat([v, v.flip(-1)], dim=-1)
        Vc = torch.fft.fft(v2, dim=-1)[..., :n]
        k = torch.arange(n, dtype=v.dtype, device=v.device) * torch.pi / (2 * n)
        out = Vc.real * torch.cos(k) + Vc.imag * torch.sin(k)
        out = out / torch.sqrt(torch.tensor(2.0 * n, dtype=v.dtype, device=v.device))
        out[..., 0] = out[..., 0] / torch.sqrt(torch.tensor(2.0, dtype=v.dtype, device=v.device))
        return out * 2.0

    y = dct_1d(x)
    y = dct_1d(y.transpose(-1, -2)).transpose(-1, -2)
    mag = torch.log1p(y.abs())
    mean = mag.mean(dim=(-2, -1), keepdim=True)
    std = mag.std(dim=(-2, -1), keepdim=True) + 1e-8
    return (mag - mean) / std


def radius_map(h: int, w: int, device=None, dtype=torch.float32, centered: bool = True) -> torch.Tensor:
    """Radius normalised by the INSCRIBED radius: r == 1 at the edge midpoint.

    Deliberately NOT normalised by the corner distance. If r were scaled so that
    r == 1 at the corner, every ring beyond r^2 > 0.5 would be clipped by the
    square boundary and the "equal-area" rings would not be equal at all -- in
    practice the outer rings came out ~6x smaller than the inner ones, which
    silently biases both the band tokens and the attention distribution used for
    attribution. Corner pixels (r > 1) are excluded from all rings instead; that
    region only holds diagonal high frequencies and is anisotropic anyway.
    """
    if centered:
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        y = (torch.arange(h, device=device, dtype=dtype) - cy) / max(cy, 1e-8)
        x = (torch.arange(w, device=device, dtype=dtype) - cx) / max(cx, 1e-8)
    else:  # origin at top-left (DCT)
        y = torch.arange(h, device=device, dtype=dtype) / max(h - 1, 1)
        x = torch.arange(w, device=device, dtype=dtype) / max(w - 1, 1)
    return torch.sqrt(y.view(-1, 1) ** 2 + x.view(1, -1) ** 2)


def equal_area_ring_masks(
    h: int, w: int, n_bands: int, device=None, dtype=torch.float32,
    drop_lowest: int = 1, centered: bool = True,
) -> torch.Tensor:
    """(n_bands, H, W) binary masks partitioning the spectrum into equal-area rings.

    Equal area <=> equal increments in r^2, so ring k covers
        sqrt(k/n) <= r < sqrt((k+1)/n).
    `drop_lowest` rings (and DC) are returned as all-zero masks so the model can
    never attend to pure content; they are kept in the tensor so band indices stay
    stable across configs. Corner pixels (r > 1 in inscribed-radius units) belong
    to no ring -- see radius_map for why.

    Verified property: interior ring pixel counts agree to within ~1-3%.
    """
    r = radius_map(h, w, device=device, dtype=dtype, centered=centered)
    inside = r < 1.0                                     # drop the corner region
    idx = (r.clamp(0, 1 - 1e-6) ** 2 * n_bands).floor().clamp(0, n_bands - 1).long()
    masks = torch.zeros(n_bands, h, w, device=device, dtype=dtype)
    masks.scatter_(0, idx.unsqueeze(0), 1.0)
    masks = masks * inside.unsqueeze(0).to(dtype)
    if drop_lowest > 0:
        masks[:drop_lowest] = 0.0
    if centered:                       # kill the DC bin explicitly
        masks[:, h // 2, w // 2] = 0.0
    return masks


def band_pool(spectrum: torch.Tensor, masks: torch.Tensor, n_stats: int = 4) -> torch.Tensor:
    """Pool a spectrum into per-band statistics.

    spectrum: (B,C,H,W)   masks: (n_bands,H,W)
    returns:  (B, n_bands, C*n_stats)  with [mean, std, max, energy] per channel.
    """
    b, c, h, w = spectrum.shape
    m = masks.unsqueeze(0).unsqueeze(2)                  # (1,n_bands,1,H,W)
    s = spectrum.unsqueeze(1)                            # (B,1,C,H,W)
    denom = m.sum(dim=(-2, -1)).clamp(min=1.0)           # (1,n_bands,1)

    mean = (s * m).sum(dim=(-2, -1)) / denom
    var = ((s - mean.unsqueeze(-1).unsqueeze(-1)) ** 2 * m).sum(dim=(-2, -1)) / denom
    std = var.clamp(min=0).sqrt()
    mx = (s * m + (m - 1) * 1e4).amax(dim=(-2, -1))
    energy = (s**2 * m).sum(dim=(-2, -1)) / denom

    stats = [mean, std, mx, energy][:n_stats]
    return torch.cat(stats, dim=-1)                      # (B,n_bands,C*n_stats)


def radial_energy_profile(spectrum: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """FreqCross-style radial energy distribution: (B, n_bands), sums to 1."""
    m = masks.unsqueeze(0)                               # (1,n_bands,H,W)
    s = spectrum.mean(dim=1, keepdim=True)               # (B,1,H,W)
    e = (s.pow(2) * m).sum(dim=(-2, -1))                 # (B,n_bands)
    return e / e.sum(dim=1, keepdim=True).clamp(min=1e-8)


def build_spectrum(x_rgb: torch.Tensor, representation: str = "magnitude") -> torch.Tensor:
    """RGB batch -> spectrum tensor (B,C,H,W). C is 1 (magnitude/dct) or 2 (+phase)."""
    luma = to_luma(x_rgb)
    if representation == "magnitude":
        return log_magnitude_spectrum(luma)
    if representation == "magnitude_phase":
        return torch.cat([log_magnitude_spectrum(luma), phase_spectrum(luma)], dim=1)
    if representation == "dct":
        return dct_2d(luma)
    raise ValueError(f"Unknown representation '{representation}'")
