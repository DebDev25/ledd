"""Stream-level causal check for the spatial-vs-frequency balance metric.

Why this exists: the project's own rollout reference (Abnar & Zuidema) shows raw
attention is a poor proxy for importance. So the [FUSE] attention split is a CLAIM.
This module tests it: zero one stream's tokens, measure how much the prediction
moves, and correlate that with the claimed balance. Report the correlation
alongside the metric -- unvalidated, the balance number should not appear in the
paper at all.
"""
from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def stream_deletion_effect(model, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Returns per-sample |Δscore| when each stream is removed, plus claimed balance."""
    base = model(x)
    p_full = torch.sigmoid(base["logit"])
    balance = base.get("balance")

    n_bands = model.frequency.n_bands
    gh, gw = base["grid"]
    n_spatial = gh * gw

    zero_bands = torch.zeros(x.shape[0], n_bands, device=x.device)
    p_no_freq = torch.sigmoid(model(x, band_mask=zero_bands)["logit"])

    zero_spatial = torch.zeros(x.shape[0], n_spatial, device=x.device)
    p_no_spatial = torch.sigmoid(model(x, spatial_token_mask=zero_spatial)["logit"])

    d_freq = (p_full - p_no_freq).abs()
    d_spatial = (p_full - p_no_spatial).abs()
    tot = (d_freq + d_spatial).clamp(min=1e-8)

    return {
        "delta_frequency": d_freq,
        "delta_spatial": d_spatial,
        "causal_balance": torch.stack([d_spatial / tot, d_freq / tot], dim=-1),
        "claimed_balance": balance,
    }


def validate_balance_metric(model, loader, device: str, max_batches: int = 20) -> Dict[str, float]:
    """Spearman correlation between claimed and causal balance.

    High correlation -> the balance metric is defensible.
    Low correlation  -> report the causal version only.
    """
    import numpy as np
    from scipy.stats import spearmanr

    claimed, causal = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["image"].to(device)
        r = stream_deletion_effect(model, x)
        if r["claimed_balance"] is None:
            return {"error": "model produced no balance metric (concat fusion?)"}
        claimed.extend(r["claimed_balance"][:, 1].cpu().tolist())     # frequency share
        causal.extend(r["causal_balance"][:, 1].cpu().tolist())

    rho, p = spearmanr(np.asarray(claimed), np.asarray(causal))
    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n": len(claimed),
        "mean_claimed_frequency_share": float(np.mean(claimed)),
        "mean_causal_frequency_share": float(np.mean(causal)),
        "verdict": "balance metric is defensible" if rho > 0.4 else
                   "claimed balance does NOT track causal effect -- report causal only",
    }
