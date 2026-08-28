"""Stream 1 -- spatial (RGB) stream: MobileViT-S via timm.

Token choice: we take the stride-16 stage, giving 14x14 = 196 tokens at 224px.
The final stride-32 stage would give 7x7 = 49, which makes the spatial heatmaps
as coarse as Grad-CAM's known weakness -- too blunt for the explainability claim.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .npr import NPRChannel


class SpatialStream(nn.Module):
    def __init__(
        self,
        name: str = "mobilevit_s",
        pretrained: bool = True,
        token_stage: int = 3,          # index into timm feature list; 3 => stride 16
        use_npr_channel: bool = False,
        image_size: int = 224,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as e:                       # pragma: no cover
            raise ImportError("timm is required for the spatial stream: pip install timm") from e

        self.npr = NPRChannel() if use_npr_channel else None
        self.backbone = timm.create_model(
            name, pretrained=pretrained, features_only=True
        )
        info = self.backbone.feature_info
        self.reductions = [f["reduction"] for f in info.info] if hasattr(info, "info") else list(info.reduction())
        self.channels = list(info.channels())

        # Prefer the stride-16 stage; fall back to the requested index.
        self.token_stage = next(
            (i for i, r in enumerate(self.reductions) if r == 16), min(token_stage, len(self.channels) - 1)
        )
        self.out_channels = self.channels[self.token_stage]
        self.final_channels = self.channels[-1]
        self.image_size = image_size

    @property
    def grid_size(self) -> int:
        return self.image_size // self.reductions[self.token_stage]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.npr is not None:
            x = self.npr(x)
        feats = self.backbone(x)
        fmap = feats[self.token_stage]                       # (B,C,h,w)
        b, c, h, w = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)             # (B, h*w, C)
        pooled = feats[-1].mean(dim=(2, 3))                  # (B, C_last)
        return {"tokens": tokens, "feature_map": fmap, "pooled": pooled, "grid": (h, w)}


class SpatialClassifier(nn.Module):
    """Stream 1 + linear head -- Stage 1 training (ablation row)."""

    def __init__(self, stream: SpatialStream, dropout: float = 0.1):
        super().__init__()
        self.stream = stream
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(stream.final_channels, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.stream(x)
        out["logit"] = self.head(out["pooled"]).squeeze(-1)
        return out
