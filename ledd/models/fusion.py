"""Fusion module -- bidirectional cross-attention + [FUSE] readout.

Guards baked in (Architecture doc S2.3):
  * `mode='concat'` reproduces the naive baseline the whole design must beat --
    DIRE found RGB+residual concat actively HURT, and FreqCross never cleanly
    showed attention fusion beats concat. This ablation is mandatory, so it is a
    config switch rather than a separate model.
  * Modality dropout stops the spatial stream (stronger in-distribution) from
    starving the frequency stream (which carries the OOD hopes).
  * The [FUSE] attention split is normalised by token count -- 196 spatial vs ~12
    band tokens would otherwise make spatial look dominant by construction.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .attention import CrossAttentionBlock, RecordedAttention


class FusionModule(nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        freq_dim: int,
        dim: int = 192,
        heads: int = 4,
        layers: int = 1,
        bidirectional: bool = True,
        modality_dropout: float = 0.25,
        mode: str = "cross_attention",
        drop: float = 0.0,
    ):
        super().__init__()
        self.mode = mode
        self.dim = dim
        self.bidirectional = bidirectional
        self.modality_dropout = modality_dropout

        self.proj_spatial = nn.Linear(spatial_dim, dim)
        self.proj_freq = nn.Linear(freq_dim, dim)

        if mode == "cross_attention":
            self.f2s = nn.ModuleList([CrossAttentionBlock(dim, heads, drop=drop) for _ in range(layers)])
            self.s2f = nn.ModuleList(
                [CrossAttentionBlock(dim, heads, drop=drop) for _ in range(layers)]
            ) if bidirectional else None

            self.fuse_token = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.fuse_token, std=0.02)
            self.readout = RecordedAttention(dim, heads)
            self.norm_q = nn.LayerNorm(dim)
            self.norm_kv = nn.LayerNorm(dim)
        elif mode == "concat":
            self.concat_proj = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU())
        else:
            raise ValueError(f"Unknown fusion mode '{mode}'")

        self.norm_out = nn.LayerNorm(dim)

    # ------------------------------------------------------------------ helpers
    def _apply_modality_dropout(self, s: torch.Tensor, f: torch.Tensor):
        """Randomly zero one stream per sample. Never both."""
        if not self.training or self.modality_dropout <= 0:
            return s, f, None
        b = s.shape[0]
        r = torch.rand(b, device=s.device)
        drop_s = (r < self.modality_dropout / 2).float().view(b, 1, 1)
        drop_f = ((r >= self.modality_dropout / 2) & (r < self.modality_dropout)).float().view(b, 1, 1)
        return s * (1 - drop_s), f * (1 - drop_f), (drop_s.view(b), drop_f.view(b))

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        spatial_tokens: torch.Tensor,      # (B, Ns, Cs)
        freq_tokens: torch.Tensor,         # (B, Nf, Cf)
        freq_cls: Optional[torch.Tensor] = None,
        spatial_pooled: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        s = self.proj_spatial(spatial_tokens)
        f = self.proj_freq(freq_tokens)
        s, f, dropped = self._apply_modality_dropout(s, f)

        if self.mode == "concat":
            # Deliberately naive: mean-pool each stream, concatenate, project.
            v = torch.cat([s.mean(1), f.mean(1)], dim=-1)
            return {"fused": self.norm_out(self.concat_proj(v)), "balance": None}

        # Freq -> Spatial: band tokens query the image. Gives band-to-region maps.
        f_upd = f
        for blk in self.f2s:
            f_upd = blk(f_upd, s)
        # Spatial -> Freq: regions pull in spectral context.
        s_upd = s
        if self.s2f is not None:
            for blk in self.s2f:
                s_upd = blk(s_upd, f)

        ns, nf = s_upd.shape[1], f_upd.shape[1]
        ctx = torch.cat([s_upd, f_upd], dim=1)
        fuse = self.fuse_token.expand(s.shape[0], -1, -1)
        out = self.readout(self.norm_q(fuse), self.norm_kv(ctx)).squeeze(1)

        balance = None
        if self.readout.attn is not None:
            a = self.readout.attn.mean(dim=1)[:, 0]          # (B, Ns+Nf)
            # Normalise by token count, else 196 vs 12 fakes a spatial win.
            sp = a[:, :ns].sum(dim=-1) / ns
            fr = a[:, ns:].sum(dim=-1) / nf
            tot = (sp + fr).clamp(min=1e-8)
            balance = torch.stack([sp / tot, fr / tot], dim=-1)   # (B, 2)

        return {
            "fused": self.norm_out(out),
            "balance": balance,
            "spatial_updated": s_upd,
            "freq_updated": f_upd,
            "n_spatial": ns,
            "n_freq": nf,
            "dropped": dropped,
        }
