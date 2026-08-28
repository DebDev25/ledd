"""Baselines we train ourselves: ResNet50, CNNDetection, NPR.

All three are wrapped to expose the same {'logit': ...} interface as LEDD so the
evaluation harness and the protocol table work unchanged. Same splits, same
preprocessing, same degradations -- otherwise the comparison means nothing.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..models.npr import npr_map


class ResNet50Baseline(nn.Module):
    """Plain ImageNet-pretrained ResNet50 fine-tuned for real/fake. The anchor."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = torchvision.models.resnet50(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)

    def forward(self, x: torch.Tensor, **kw) -> Dict[str, torch.Tensor]:
        return {"logit": self.net(x).squeeze(-1)}


class CNNDetectionBaseline(nn.Module):
    """Wang et al. 2020 recipe: ResNet50 + blur/JPEG augmentation.

    Architecturally identical to the anchor; the METHOD is the augmentation policy,
    which lives in the data pipeline (augment.enabled + blur/jpeg probabilities).
    Use configs/baselines/cnndetection.yaml, which sets their p=0.5 blur+JPEG.

    To instead run the authors' released weights, see baselines/README.md.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = ResNet50Baseline(pretrained)

    def forward(self, x: torch.Tensor, **kw) -> Dict[str, torch.Tensor]:
        return self.model(x)


class NPRBaseline(nn.Module):
    """Tan et al., CVPR 2024 -- NPR map -> small ResNet-block CNN (~1.4M params).

    The most relevant baseline for the lightweight pillar, and cheap to train.
    Note their paper reports NO robustness testing under JPEG/blur/resize; running
    our robustness protocol on this model is a small contribution in itself.
    """

    def __init__(self, grid: int = 2, width: int = 64):
        super().__init__()
        self.grid = grid

        def block(cin, cout, stride=2):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.body = nn.Sequential(block(width, width), block(width, width * 2),
                                  block(width * 2, width * 2))
        self.head = nn.Linear(width * 2, 1)

    def forward(self, x: torch.Tensor, **kw) -> Dict[str, torch.Tensor]:
        n = npr_map(x, self.grid)
        h = self.body(self.stem(n)).mean(dim=(2, 3))
        return {"logit": self.head(h).squeeze(-1)}


BASELINES = {
    "resnet50": ResNet50Baseline,
    "cnndetection": CNNDetectionBaseline,
    "npr": NPRBaseline,
}


def build_baseline(name: str, **kw) -> nn.Module:
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline '{name}'. Available: {sorted(BASELINES)}")
    return BASELINES[name](**kw)
