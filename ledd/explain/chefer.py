"""Relevance propagation for transformer attention (Chefer, Gur & Wolf).

Two papers, two jobs:
  * CVPR 2021 "Transformer Interpretability Beyond Attention Visualization"
    (arXiv:2012.09838) -- within-stream self-attention relevance. Improves on plain
    rollout by weighting attention with its gradient and keeping only positive
    contributions.
  * ICCV 2021 "Generic Attention-model Explainability for Interpreting Bi-Modal and
    Encoder-Decoder Transformers" (arXiv:2103.15679) -- propagation THROUGH
    cross-attention, which plain rollout does not handle. This is what our fusion
    module needs.

Core rule for both: use the gradient-weighted attention
    A_bar = E_h[ (grad_A * A)^+ ]
and propagate relevance with residual-aware matrix products.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from ..models.attention import RecordedAttention


def gradient_weighted_attention(attn: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """A_bar = mean_heads( relu(grad * attn) ).  attn/grad: (B, heads, Nq, Nk)."""
    return (grad * attn).clamp(min=0).mean(dim=1)


def _norm_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.sum(dim=-1, keepdim=True).clamp(min=eps)


def self_attention_relevance(blocks: List[RecordedAttention]) -> Optional[torch.Tensor]:
    """CVPR-2021 rule for a stack of self-attention layers.

    R <- R + A_bar @ R,  starting from identity (the residual stream).
    Returns (B, N, N); row i is the relevance of every token to token i.
    """
    if not blocks or blocks[0].attn is None:
        return None
    b, _, n, _ = blocks[0].attn.shape
    device = blocks[0].attn.device
    R = torch.eye(n, device=device).unsqueeze(0).expand(b, -1, -1).clone()
    for blk in blocks:
        if blk.attn is None:
            continue
        grad = blk.attn_grad if blk.attn_grad is not None else torch.ones_like(blk.attn)
        A = gradient_weighted_attention(blk.attn, grad)
        A = _norm_rows(A + torch.eye(n, device=device).unsqueeze(0))   # residual
        R = R + torch.bmm(A, R)
    return R


def cross_attention_relevance(
    cross_block: RecordedAttention,
    R_q_self: Optional[torch.Tensor] = None,
    R_kv_self: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """ICCV-2021 rule for a cross-attention layer.

    R_qk = R_q_self^T @ A_bar @ R_kv_self
    i.e. relevance flows: query-side self-context -> cross map -> key-side context.
    Returns (B, Nq, Nk).
    """
    if cross_block.attn is None:
        return None
    grad = cross_block.attn_grad if cross_block.attn_grad is not None else torch.ones_like(cross_block.attn)
    A = gradient_weighted_attention(cross_block.attn, grad)      # (B, Nq, Nk)
    if R_q_self is not None:
        A = torch.bmm(R_q_self.transpose(1, 2), A)
    if R_kv_self is not None:
        A = torch.bmm(A, R_kv_self)
    return A


def collect_attention_blocks(module) -> List[RecordedAttention]:
    return [m for m in module.modules() if isinstance(m, RecordedAttention)]
