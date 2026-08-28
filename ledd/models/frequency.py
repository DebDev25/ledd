"""Stream 2 -- frequency stream with radial band tokens.

Why band tokens instead of a CNN over the raw spectrum
-----------------------------------------------------
GenImage and UniversalFakeDetect both show diffusion images lack the obvious GAN
grid artifacts; NPR (Tan et al. 2024) shows raw spectral features overfit to the
training source. FIRE (CVPR 2025) localises the generalisable signal in the
mid-band. So we make the band structure explicit:

  spectrum -> equal-area rings -> one token per ring -> tiny transformer + CLS

Consequences that fall out of the architecture rather than being bolted on:
  * CLS->band attention IS the frequency-band attribution (no post-hoc masking);
  * the band-entropy regulariser has a well-defined distribution to act on;
  * band deletion/insertion is exact (zero a ring), avoiding RISE's caveat that
    blur-based deletion leaves exploitable low-frequency information.

Future work (deferred): angular sectors for orientation-specific artifacts.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .attention import SelfAttentionBlock
from .fft import band_pool, build_spectrum, equal_area_ring_masks, radial_energy_profile


class BandEncoder(nn.Module):
    """Shared MLP mapping per-band statistics -> band token embedding."""

    def __init__(self, in_dim: int, dim: int, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrequencyStream(nn.Module):
    """(B,3,H,W) RGB -> band tokens (B, n_bands, dim) + CLS (B, dim).

    Total parameters stay well under 1M at the default settings.
    """

    def __init__(
        self,
        n_bands: int = 12,
        drop_lowest: int = 1,
        representation: str = "magnitude",
        embed_dim: int = 160,
        depth: int = 2,
        heads: int = 4,
        use_radial_mlp: bool = True,
        n_stats: int = 4,
        drop: float = 0.0,
        image_size: int = 224,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.drop_lowest = drop_lowest
        self.representation = representation
        self.embed_dim = embed_dim
        self.use_radial_mlp = use_radial_mlp
        self.n_stats = n_stats
        self.centered = representation != "dct"

        n_ch = 2 if representation == "magnitude_phase" else 1
        self.band_encoder = BandEncoder(n_ch * n_stats, embed_dim, drop)

        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.band_pos = nn.Parameter(torch.zeros(1, n_bands, embed_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.band_pos, std=0.02)

        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(embed_dim, heads, drop=drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # FreqCross-style radial energy MLP: nearly free, and their ablation showed
        # the radial-energy modality carried real signal.
        if use_radial_mlp:
            self.radial_mlp = nn.Sequential(
                nn.Linear(n_bands, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
            )

        self._mask_cache: Dict[tuple, torch.Tensor] = {}
        self.image_size = image_size

    # ------------------------------------------------------------------ masks
    def ring_masks(self, h: int, w: int, device, dtype) -> torch.Tensor:
        key = (h, w, str(device), str(dtype))
        if key not in self._mask_cache:
            self._mask_cache[key] = equal_area_ring_masks(
                h, w, self.n_bands, device=device, dtype=dtype,
                drop_lowest=self.drop_lowest, centered=self.centered,
            )
        return self._mask_cache[key]

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x_rgb: Optional[torch.Tensor] = None,
        spectrum: Optional[torch.Tensor] = None,
        band_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """band_mask: (B, n_bands) multiplicative mask for band-deletion experiments."""
        if spectrum is None:
            assert x_rgb is not None, "provide x_rgb or spectrum"
            spectrum = build_spectrum(x_rgb, self.representation)

        b, _, h, w = spectrum.shape
        masks = self.ring_masks(h, w, spectrum.device, spectrum.dtype)

        feats = band_pool(spectrum, masks, self.n_stats)      # (B, n_bands, C*n_stats)
        tokens = self.band_encoder(feats) + self.band_pos     # (B, n_bands, dim)

        if band_mask is not None:
            tokens = tokens * band_mask.unsqueeze(-1).to(tokens.dtype)

        if self.use_radial_mlp:
            prof = radial_energy_profile(spectrum, masks)     # (B, n_bands)
            if band_mask is not None:
                prof = prof * band_mask.to(prof.dtype)
            cls = self.cls.expand(b, -1, -1) + self.radial_mlp(prof).unsqueeze(1)
        else:
            cls = self.cls.expand(b, -1, -1)

        seq = torch.cat([cls, tokens], dim=1)                 # (B, 1+n_bands, dim)
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)

        return {
            "cls": seq[:, 0],
            "tokens": seq[:, 1:],
            "spectrum": spectrum,
            "masks": masks,
        }

    # -------------------------------------------------------------- attribution
    def band_attention(self, head_average: bool = True) -> Optional[torch.Tensor]:
        """CLS -> band attention from the LAST block: (B, n_bands), renormalised.

        This is the intrinsic frequency-band attribution AND the distribution the
        band-entropy regulariser acts on. Requires attention recording to be on.
        """
        last = self.blocks[-1].attn
        if last.attn is None:
            return None
        a = last.attn                                  # (B, heads, N, N)
        a = a.mean(dim=1) if head_average else a[:, 0]
        cls_to_bands = a[:, 0, 1:]                     # drop CLS->CLS
        return cls_to_bands / cls_to_bands.sum(dim=-1, keepdim=True).clamp(min=1e-8)


class FrequencyClassifier(nn.Module):
    """Stream 2 + linear head -- used standalone in Stage 2 (ablation row)."""

    def __init__(self, stream: FrequencyStream, dropout: float = 0.1):
        super().__init__()
        self.stream = stream
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(stream.embed_dim, 1)
        )

    def forward(self, x_rgb: torch.Tensor, band_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = self.stream(x_rgb, band_mask=band_mask)
        out["logit"] = self.head(out["cls"]).squeeze(-1)
        return out
