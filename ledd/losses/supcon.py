"""Supervised contrastive loss (Khosla et al., NeurIPS 2020) -- L_out formulation.

Two classes only: real (0) and fake (1). Pulling ALL reals together and ALL fakes
together regardless of which generator produced the fake is the mechanism aimed at
cross-generator collapse.

L_out (sum outside the log) is used deliberately -- the paper proves the
1/|P(i)| normalisation removes a positive bias in the gradient that makes the
L_in variant measurably worse.

Honest framing for the paper: SupCon demonstrably improves in-distribution accuracy
and corruption robustness, but showed NO transfer gain in the original work. Its
cross-generator benefit here is a hypothesis the ablation tests.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .queue import FeatureQueue


def supcon_loss(
    feats: torch.Tensor,                 # (B, D) L2-normalised
    labels: torch.Tensor,                # (B,)
    queue_feats: Optional[torch.Tensor] = None,
    queue_labels: Optional[torch.Tensor] = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    device = feats.device
    b = feats.shape[0]
    labels = labels.view(-1).long()

    if queue_feats is not None and queue_feats.numel() > 0:
        cand = torch.cat([feats, queue_feats.to(device).to(feats.dtype)], dim=0)
        cand_labels = torch.cat([labels, queue_labels.to(device).long()], dim=0)
    else:
        cand, cand_labels = feats, labels

    logits = (feats @ cand.T) / temperature                     # (B, N)

    # mask out self-comparisons (first b columns are the batch itself)
    self_mask = torch.zeros_like(logits, dtype=torch.bool)
    self_mask[:, :b] = torch.eye(b, device=device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, float("-inf"))

    pos_mask = (labels.view(-1, 1) == cand_labels.view(1, -1)) & ~self_mask

    # log-softmax over all candidates (the denominator), then average over positives
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    n_pos = pos_mask.sum(dim=1)
    valid = n_pos > 0
    if not valid.any():
        return feats.sum() * 0.0
    mean_log_prob_pos = (log_prob.masked_fill(~pos_mask, 0.0) * pos_mask).sum(dim=1)[valid] / n_pos[valid]
    return -mean_log_prob_pos.mean()


class SupConCriterion(nn.Module):
    """SupCon with an integrated queue, warm-up and linear ramp.

    The queue must be primed before the loss is meaningful: for the first
    `warmup_steps` the queue holds random/stale vectors, so the loss is disabled
    and only enqueueing happens. Then lambda ramps linearly over `ramp_steps`.
    """

    def __init__(
        self,
        dim: int = 128,
        temperature: float = 0.1,
        queue_size: int = 8192,
        warmup_steps: int = 500,
        ramp_steps: int = 1000,
    ):
        super().__init__()
        self.temperature = temperature
        self.warmup_steps = warmup_steps
        self.ramp_steps = max(ramp_steps, 1)
        self.queue = FeatureQueue(dim, queue_size) if queue_size > 0 else None
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

    def weight_scale(self) -> float:
        s = int(self.step.item())
        if s < self.warmup_steps:
            return 0.0
        return min(1.0, (s - self.warmup_steps) / self.ramp_steps)

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        qf = ql = None
        if self.queue is not None:
            qf, ql = self.queue.get()
        scale = self.weight_scale()
        loss = (supcon_loss(feats, labels, qf, ql, self.temperature)
                if scale > 0 else feats.sum() * 0.0)
        if self.queue is not None:
            self.queue.enqueue(feats, labels)
        if self.training:
            self.step += 1
        return loss * scale
