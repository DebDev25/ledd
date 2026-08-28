"""LEDD -- the full dual-stream detector.

  RGB --> MobileViT-S ------ 196 tokens --\
                                            >-- cross-attention fusion --> [FUSE] --> head
  RGB --> FFT rings --> band transformer --/

~7M parameters total. Everything the explainability layer needs (band attention,
[FUSE] balance, recorded attention maps) comes out of the forward pass.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .frequency import FrequencyStream
from .fusion import FusionModule
from .spatial import SpatialStream


class ProjectionHead(nn.Module):
    """SupCon projection head. Discarded at inference -> zero deployment cost,
    so the 'lightweight' claim is measured without it."""

    def __init__(self, in_dim: int, hidden: int = 192, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=-1)


class LEDDDetector(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        mcfg = cfg["model"]
        img = cfg.get("data", {}).get("image_size", 224)

        self.spatial = SpatialStream(
            name=mcfg["spatial"].get("name", "mobilevit_s"),
            pretrained=mcfg["spatial"].get("pretrained", True),
            token_stage=mcfg["spatial"].get("token_stage", 3),
            use_npr_channel=mcfg["spatial"].get("use_npr_channel", False),
            image_size=img,
        )
        fq = mcfg["frequency"]
        self.frequency = FrequencyStream(
            n_bands=fq.get("n_bands", 12),
            drop_lowest=fq.get("drop_lowest", 1),
            representation=fq.get("representation", "magnitude"),
            embed_dim=fq.get("embed_dim", 160),
            depth=fq.get("depth", 2),
            heads=fq.get("heads", 4),
            use_radial_mlp=fq.get("use_radial_mlp", True),
            image_size=img,
        )
        fu = mcfg["fusion"]
        self.fusion = FusionModule(
            spatial_dim=self.spatial.out_channels,
            freq_dim=self.frequency.embed_dim,
            dim=fu.get("dim", 192),
            heads=fu.get("heads", 4),
            layers=fu.get("layers", 1),
            bidirectional=fu.get("bidirectional", True),
            modality_dropout=fu.get("modality_dropout", 0.25),
            mode=fu.get("mode", "cross_attention"),
        )
        hd = mcfg.get("head", {})
        d = fu.get("dim", 192)
        self.head = nn.Sequential(
            nn.Dropout(hd.get("dropout", 0.1)),
            nn.Linear(d, hd.get("hidden", d)),
            nn.GELU(),
            nn.Linear(hd.get("hidden", d), 1),
        )
        pj = mcfg.get("projection", {})
        self.projection = ProjectionHead(d, pj.get("hidden", 192), pj.get("dim", 128))

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        x: torch.Tensor,
        band_mask: Optional[torch.Tensor] = None,
        spatial_token_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        s = self.spatial(x)
        f = self.frequency(x, band_mask=band_mask)

        st = s["tokens"]
        if spatial_token_mask is not None:               # for stream/region deletion
            st = st * spatial_token_mask.unsqueeze(-1).to(st.dtype)

        fu = self.fusion(st, f["tokens"], freq_cls=f["cls"], spatial_pooled=s["pooled"])
        logit = self.head(fu["fused"]).squeeze(-1)

        out = {
            "logit": logit,
            "fused": fu["fused"],
            "balance": fu["balance"],
            "band_attention": self.frequency.band_attention(),
            "grid": s["grid"],
        }
        if return_embeddings:
            out["proj"] = self.projection(fu["fused"])
        return out

    # -------------------------------------------------------------- param groups
    def param_groups(self, lr_backbone: float, lr_new: float, weight_decay: float = 0.05):
        """Pretrained streams get a small LR; freshly-initialised parts get the big one.
        Norm/bias params are excluded from weight decay."""
        backbone, new, no_decay = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            elif name.startswith("spatial."):
                backbone.append(p)
            else:
                new.append(p)
        return [
            {"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": new, "lr": lr_new, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr_new, "weight_decay": 0.0},
        ]

    def freeze_streams(self, freeze: bool = True) -> None:
        for m in (self.spatial, self.frequency):
            for p in m.parameters():
                p.requires_grad = not freeze

    def load_stream_checkpoints(self, spatial_ckpt: Optional[str] = None,
                                frequency_ckpt: Optional[str] = None,
                                map_location: str = "cpu") -> Dict[str, Any]:
        """Load Stage 1 / Stage 2 weights into the streams, ignoring their heads."""
        report = {}
        for tag, path, module in (("spatial", spatial_ckpt, self.spatial),
                                  ("frequency", frequency_ckpt, self.frequency)):
            if not path:
                continue
            ck = torch.load(path, map_location=map_location, weights_only=False)
            sd = ck.get("model", ck)
            prefix = "stream."
            sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            missing = module.load_state_dict(sub or sd, strict=False)
            report[tag] = {"missing": list(missing.missing_keys), "unexpected": list(missing.unexpected_keys)}
        return report


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
