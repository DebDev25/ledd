"""Attention blocks that expose their weights and support gradient hooks.

Explainability (Chefer et al., CVPR/ICCV 2021) needs BOTH the attention map and
the gradient of the output w.r.t. that map. Standard nn.MultiheadAttention gives
neither conveniently, so we implement small explicit blocks.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecordedAttention(nn.Module):
    """Multi-head attention that stores its attention map and (optionally) its grad."""

    def __init__(self, dim: int, heads: int = 4, dim_kv: Optional[int] = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        dim_kv = dim_kv or dim

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim_kv, dim)
        self.v = nn.Linear(dim_kv, dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn: Optional[torch.Tensor] = None       # (B, heads, Nq, Nk)
        self.attn_grad: Optional[torch.Tensor] = None
        self.store_attn = False
        self.store_grad = False

    def _save_grad(self, grad: torch.Tensor) -> None:
        self.attn_grad = grad.detach()

    def forward(self, q_in: torch.Tensor, kv_in: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        kv_in = q_in if kv_in is None else kv_in
        b, nq, _ = q_in.shape
        nk = kv_in.shape[1]

        q = self.q(q_in).view(b, nq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(kv_in).view(b, nk, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(kv_in).view(b, nk, self.heads, self.head_dim).transpose(1, 2)

        logits = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:               # (B, Nk) True = drop
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = logits.softmax(dim=-1)

        if self.store_attn:
            self.attn = attn if self.store_grad else attn.detach()
        if self.store_grad and attn.requires_grad:
            attn.register_hook(self._save_grad)

        out = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(b, nq, -1)
        return self.proj_drop(self.proj(out))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        hidden = hidden or dim * 2
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = RecordedAttention(dim, heads, proj_drop=drop)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class CrossAttentionBlock(nn.Module):
    """One direction of cross-attention: `x` queries `ctx`."""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        self.norm_q, self.norm_kv, self.norm_mlp = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = RecordedAttention(dim, heads, proj_drop=drop)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor,
                ctx_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm_q(x), self.norm_kv(ctx), key_padding_mask=ctx_mask)
        return x + self.mlp(self.norm_mlp(x))


def set_attention_recording(module: nn.Module, store_attn: bool = True, store_grad: bool = False) -> None:
    """Toggle recording on every RecordedAttention in a model.

    Recording is OFF during training (it holds a B x heads x Nq x Nk tensor per
    layer, which is wasted memory on a free-tier GPU) and ON for explainability.
    """
    for m in module.modules():
        if isinstance(m, RecordedAttention):
            m.store_attn = store_attn
            m.store_grad = store_grad
            if not store_attn:
                m.attn = None
                m.attn_grad = None
