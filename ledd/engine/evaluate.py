"""Evaluation harness implementing the protocol table (Architecture doc S7.2)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

from ..data.dataset import build_loader
from ..data.splits import build_index, load_splits
from .metrics import classification_metrics, per_generator_metrics


@torch.no_grad()
def collect_predictions(model, loader, device: str, amp: bool = True) -> Dict[str, list]:
    model.eval()
    logits, labels, gens, balances = [], [], [], []
    for batch in tqdm(loader, desc="predict", leave=False):
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], enabled=amp and device != "cpu"):
            out = model(x)
        logits.extend(out["logit"].float().cpu().tolist())
        labels.extend(batch["label"].tolist())
        gens.extend(batch["generator"].tolist())
        if out.get("balance") is not None:
            balances.extend(out["balance"].float().cpu().tolist())
    return {"logits": logits, "labels": labels, "generators": gens, "balance": balances}


def evaluate_split(model, items, device: str, batch_size: int = 128,
                   degradation: Optional[str] = None, num_workers: int = 4,
                   amp: bool = True) -> Dict[str, Any]:
    loader = build_loader(items, batch_size, train=False, num_workers=num_workers,
                          degradation=degradation)
    preds = collect_predictions(model, loader, device, amp)
    res = classification_metrics(preds["logits"], preds["labels"])
    res["per_generator"] = per_generator_metrics(preds["logits"], preds["labels"], preds["generators"])
    if preds["balance"]:
        import numpy as np

        b = np.asarray(preds["balance"])
        res["mean_balance_spatial"] = float(b[:, 0].mean())
        res["mean_balance_frequency"] = float(b[:, 1].mean())
    return res


def run_protocol(
    model,
    cfg: Dict[str, Any],
    splits_path: str,
    ood_items: Optional[Sequence] = None,
    degradations: Optional[List[str]] = None,
    out_path: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Runs every row of the evaluation table that does not need a separate model."""
    splits = load_splits(splits_path)
    bs = cfg["eval"]["batch_size"]
    nw = cfg["data"].get("num_workers", 4)
    amp = cfg.get("amp", True)
    degradations = degradations or cfg["eval"].get("degradations", ["none"])

    results: Dict[str, Any] = {}
    results["in_distribution"] = evaluate_split(model, splits["val_indist"], device, bs, None, nw, amp)
    results["cross_generator_val"] = evaluate_split(model, splits["val_generator"], device, bs, None, nw, amp)
    results["cross_generator_test"] = evaluate_split(model, splits["test_generator"], device, bs, None, nw, amp)

    if ood_items:
        results["cross_model_family"] = evaluate_split(model, ood_items, device, bs, None, nw, amp)

    robustness = {}
    target = ood_items if ood_items else splits["test_generator"]
    for d in degradations:
        robustness[d] = evaluate_split(model, target, device, bs, d, nw, amp)
    results["robustness"] = robustness

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    return results


def load_ood_items(ood_root: str) -> List:
    """OOD archive layout: <ood_root>/<generator>/{real,fake}/*.png"""
    gens = [d for d in sorted(os.listdir(ood_root)) if os.path.isdir(os.path.join(ood_root, d))]
    return build_index(ood_root, gens)
