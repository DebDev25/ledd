"""MoCo-style feature queue.

Free-tier batches (64-128) are far too small for SupCon, which benefits from many
negatives, and gradient accumulation does NOT add in-batch negatives. The queue
stores recent projected embeddings (detached) so each anchor sees thousands of
negatives at batch-size cost.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class FeatureQueue(nn.Module):
    def __init__(self, dim: int = 128, size: int = 8192):
        super().__init__()
        self.size = size
        self.register_buffer("feats", torch.randn(size, dim))
        self.feats = nn.functional.normalize(self.feats, dim=-1)
        self.register_buffer("labels", torch.full((size,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        feats, labels = feats.detach(), labels.detach().long()
        n = feats.shape[0]
        ptr = int(self.ptr.item())
        idx = (torch.arange(n, device=feats.device) + ptr) % self.size
        self.feats[idx] = feats.to(self.feats.dtype)
        self.labels[idx] = labels
        self.ptr[0] = (ptr + n) % self.size
        self.filled[0] = min(int(self.filled.item()) + n, self.size)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        n = int(self.filled.item())
        return self.feats[:n], self.labels[:n]

    @property
    def is_ready(self) -> bool:
        return int(self.filled.item()) >= self.size // 2

    def __len__(self) -> int:
        return int(self.filled.item())
