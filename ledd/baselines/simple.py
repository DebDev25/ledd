"""Baselines we train ourselves: ResNet50, CNNDetection, NPR, UnivFD.

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




class UnivFDBaseline(nn.Module):
    """Ojha et al., CVPR 2023 -- linear probe on a FROZEN CLIP ViT-L/14 (~304M params).

    The heavyweight end of the comparison, and the one that makes the lightweight
    claim meaningful: this is roughly 54x our model with no task-specific training
    of the representation at all.

    Two details that matter for a fair comparison:

    * CLIP expects its own channel statistics, not ImageNet's. Our dataloader
      normalises with ImageNet mean/std, so we invert that and re-apply CLIP's
      inside forward(). Feeding ImageNet-normalised tensors to CLIP silently costs
      several AUC points and looks like the baseline being weak.
    * The backbone is frozen and kept in eval mode even when the trainer calls
      .train(), so its BatchNorm/LayerNorm statistics and dropout never move. Only
      the linear head learns, which is what "linear probe" means.
    """

    # CLIP's normalisation constants (OpenAI).
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14",
                 feature_dim: int = 768):
        super().__init__()
        try:
            from transformers import CLIPVisionModelWithProjection

            self.clip = CLIPVisionModelWithProjection.from_pretrained(model_name)
            self._backend = "hf"
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "UnivFD needs CLIP ViT-L/14. `pip install transformers` (Kaggle has it), "
                f"or edit this class to use open_clip. Original error: {e}"
            )
        for p_ in self.clip.parameters():
            p_.requires_grad = False
        self.clip.eval()
        self.head = nn.Linear(feature_dim, 1)

        def _buf(name, vals):
            self.register_buffer(name, torch.tensor(vals).view(1, 3, 1, 1), persistent=False)
        _buf("_in_mean", self.IMAGENET_MEAN); _buf("_in_std", self.IMAGENET_STD)
        _buf("_cl_mean", self.CLIP_MEAN);     _buf("_cl_std", self.CLIP_STD)

    def train(self, mode: bool = True):
        """Keep the frozen encoder in eval mode regardless of the trainer."""
        super().train(mode)
        self.clip.eval()
        return self

    def forward(self, x: torch.Tensor, **kw) -> Dict[str, torch.Tensor]:
        img = x * self._in_std + self._in_mean           # undo ImageNet normalisation
        z = (img - self._cl_mean) / self._cl_std          # apply CLIP's
        with torch.no_grad():
            feat = self.clip(pixel_values=z).image_embeds
        return {"logit": self.head(feat.float()).squeeze(-1)}


BASELINES = {
    "resnet50": ResNet50Baseline,
    "cnndetection": CNNDetectionBaseline,
    "npr": NPRBaseline,
    "univfd": UnivFDBaseline,
}


def build_baseline(name: str, **kw) -> nn.Module:
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline '{name}'. Available: {sorted(BASELINES)}")
    return BASELINES[name](**kw)
