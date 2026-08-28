"""Checkpoint save/resume.

Free-tier sessions are killed without warning, so we save model + optimizer +
scheduler + scaler + epoch + RNG every epoch and resume transparently.
"""
from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    scaler: Any = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: float = -1.0,
    extra: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a killed session never leaves a corrupt ckpt
    if is_best:
        shutil.copyfile(path, os.path.join(os.path.dirname(path), "best.pth"))


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    missing = model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    rng = ckpt.get("rng") or {}
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
    return {
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "best_metric": ckpt.get("best_metric", -1.0),
        "extra": ckpt.get("extra", {}),
        "missing": missing,
    }


def find_resume(ckpt_dir: str, mode: str = "auto") -> Optional[str]:
    if mode == "none":
        return None
    if mode not in ("auto",):
        return mode if os.path.exists(mode) else None
    last = os.path.join(ckpt_dir, "last.pth")
    return last if os.path.exists(last) else None
