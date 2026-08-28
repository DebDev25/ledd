"""Causal faithfulness metrics (Petsiuk et al., RISE, BMVC 2018).

deletion AUC  -- remove the most-important evidence first; a faithful explanation
                 makes the score fall FAST, so LOWER is better.
insertion AUC -- add the most-important evidence first; HIGHER is better.

Two design cautions carried over from RISE:
  1. AUC values depend on the removal strategy. We default to a CONSTANT fill
     (per-image mean) rather than blur, because RISE showed classifiers reconstruct
     detail from the low-frequency information blur leaves behind -- and our model
     has a frequency stream, which would exploit exactly that. The strategy is
     recorded in the returned dict so numbers are comparable across runs.
  2. Frequency-band deletion is EXACT (a ring is zeroed, not approximated), so the
     band metrics do not inherit the pixel-space ambiguity at all.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import torch

from ..engine.metrics import curve_auc


def _score(model, x: torch.Tensor, **kw) -> torch.Tensor:
    with torch.no_grad():
        return torch.sigmoid(model(x, **kw)["logit"])


def deletion_insertion_pixels(
    model,
    x: torch.Tensor,
    saliency: torch.Tensor,
    steps: int = 32,
    fill: str = "mean",
    batch_eval: int = 32,
) -> Dict[str, float]:
    """Pixel-space deletion/insertion. x:(B,3,H,W), saliency:(B,H,W)."""
    b, c, h, w = x.shape
    n = h * w
    order = saliency.flatten(1).argsort(dim=1, descending=True)      # most important first

    if fill == "mean":
        base = x.mean(dim=(2, 3), keepdim=True).expand_as(x).clone()
    elif fill == "zero":
        base = torch.zeros_like(x)
    elif fill == "blur":
        import torch.nn.functional as F

        k = torch.ones(c, 1, 11, 11, device=x.device) / 121.0
        base = F.conv2d(F.pad(x, (5, 5, 5, 5), mode="reflect"), k, groups=c)
    else:
        raise ValueError(fill)

    chunk = max(1, n // steps)
    del_scores, ins_scores = [], []
    deleted, inserted = x.clone(), base.clone()

    del_scores.append(_score(model, deleted).cpu())
    ins_scores.append(_score(model, inserted).cpu())

    for s in range(steps):
        idx = order[:, s * chunk:(s + 1) * chunk]
        if idx.numel() == 0:
            break
        flat_del = deleted.view(b, c, n)
        flat_ins = inserted.view(b, c, n)
        flat_base = base.view(b, c, n)
        flat_src = x.view(b, c, n)
        gather = idx.unsqueeze(1).expand(-1, c, -1)
        flat_del.scatter_(2, gather, flat_base.gather(2, gather))
        flat_ins.scatter_(2, gather, flat_src.gather(2, gather))
        del_scores.append(_score(model, deleted).cpu())
        ins_scores.append(_score(model, inserted).cpu())

    d = torch.stack(del_scores, dim=1)      # (B, steps+1)
    i = torch.stack(ins_scores, dim=1)
    return {
        "deletion_auc": float(torch.stack([torch.tensor(curve_auc(r.tolist())) for r in d]).mean()),
        "insertion_auc": float(torch.stack([torch.tensor(curve_auc(r.tolist())) for r in i]).mean()),
        "fill": fill,
        "steps": steps,
        "deletion_curve": d.mean(dim=0).tolist(),
        "insertion_curve": i.mean(dim=0).tolist(),
    }


def deletion_insertion_bands(
    model,
    x: torch.Tensor,
    band_attribution: torch.Tensor,
    n_bands: Optional[int] = None,
) -> Dict[str, float]:
    """Frequency-band deletion/insertion -- the novel attribution channel.

    Exact by construction: a band is removed by zeroing its token (and its radial
    energy entry), not by any approximate perturbation.
    """
    b, nb = band_attribution.shape
    n_bands = n_bands or nb
    order = band_attribution.argsort(dim=1, descending=True)

    del_mask = torch.ones(b, nb, device=x.device)
    ins_mask = torch.zeros(b, nb, device=x.device)
    del_scores = [_score(model, x, band_mask=del_mask).cpu()]
    ins_scores = [_score(model, x, band_mask=ins_mask).cpu()]

    for k in range(nb):
        idx = order[:, k:k + 1]
        del_mask = del_mask.scatter(1, idx, 0.0)
        ins_mask = ins_mask.scatter(1, idx, 1.0)
        del_scores.append(_score(model, x, band_mask=del_mask).cpu())
        ins_scores.append(_score(model, x, band_mask=ins_mask).cpu())

    d = torch.stack(del_scores, dim=1)
    i = torch.stack(ins_scores, dim=1)
    return {
        "band_deletion_auc": float(torch.stack([torch.tensor(curve_auc(r.tolist())) for r in d]).mean()),
        "band_insertion_auc": float(torch.stack([torch.tensor(curve_auc(r.tolist())) for r in i]).mean()),
        "n_bands": nb,
        "deletion_curve": d.mean(dim=0).tolist(),
        "insertion_curve": i.mean(dim=0).tolist(),
    }


def random_baseline_maps(shape, device=None) -> torch.Tensor:
    """Random saliency -- the control every faithfulness table needs.
    If your attention maps do not beat this, they are not explanations."""
    return torch.rand(*shape, device=device)
