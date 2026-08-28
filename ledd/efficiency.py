"""Efficiency measurement -- the evidence for the 'lightweight' pillar.

Reports parameters, FLOPs and latency on CPU and GPU. The SupCon projection head
is excluded from the deployment numbers because it is discarded at inference.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import torch


def count_params(model: torch.nn.Module, exclude_prefixes=("projection",)) -> Dict[str, int]:
    total = trainable = deploy = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
        if not any(name.startswith(pref) for pref in exclude_prefixes):
            deploy += p.numel()
    return {"total": total, "trainable": trainable, "deployment": deploy}


def measure_flops(model: torch.nn.Module, input_size=(1, 3, 224, 224)) -> Optional[float]:
    x = torch.zeros(*input_size)
    try:
        from fvcore.nn import FlopCountAnalysis

        model.eval()
        return float(FlopCountAnalysis(model, x).total())
    except Exception:
        pass
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(x,), verbose=False)
        return float(macs) * 2.0
    except Exception:
        return None


@torch.no_grad()
def measure_latency(model: torch.nn.Module, device: str = "cpu", input_size=(1, 3, 224, 224),
                    warmup: int = 10, iters: int = 50) -> Dict[str, float]:
    model = model.to(device).eval()
    x = torch.zeros(*input_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return {"latency_ms": dt * 1000, "throughput_img_s": input_size[0] / dt}


def full_report(model: torch.nn.Module, out_path: Optional[str] = None,
                batch_sizes=(1, 32)) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"params": count_params(model)}
    rep["flops_224"] = measure_flops(model)
    rep["latency"] = {}
    for dev in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
        for bs in batch_sizes:
            try:
                rep["latency"][f"{dev}_bs{bs}"] = measure_latency(model, dev, (bs, 3, 224, 224))
            except Exception as e:      # OOM on a small GPU is informative, not fatal
                rep["latency"][f"{dev}_bs{bs}"] = {"error": str(e)}
    if out_path:
        with open(out_path, "w") as f:
            json.dump(rep, f, indent=2)
    return rep
