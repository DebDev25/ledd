"""Three-stage trainer with free-tier survival built in.

Stages (Architecture doc S6):
  spatial    -- Stream 1 alone, BCE only
  frequency  -- Stream 2 alone, BCE only
  joint      -- both + fusion, full loss, modality dropout

Free-tier assumptions: sessions die without warning, so we checkpoint model +
optimizer + scheduler + scaler + RNG every epoch and resume transparently.
Attention recording is OFF during training (it would hold a
B x heads x Nq x Nk tensor per layer for nothing) EXCEPT when the band-entropy
term is active, which needs the band attention.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Optional

import torch
from tqdm import tqdm

from ..data.dataset import build_loader
from ..data.splits import SplitSpec, load_splits, make_splits, save_splits
from ..losses import CombinedLoss
from ..models import FrequencyClassifier, FrequencyStream, LEDDDetector, SpatialClassifier, SpatialStream
from ..models.attention import set_attention_recording
from ..utils.checkpoint import find_resume, load_checkpoint, save_checkpoint
from ..utils.logging import CSVLogger, get_logger
from ..utils.seed import set_seed
from .metrics import classification_metrics


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    stage = cfg.get("stage", "joint")
    if stage == "spatial":
        s = SpatialStream(
            name=cfg["model"]["spatial"].get("name", "mobilevit_s"),
            pretrained=cfg["model"]["spatial"].get("pretrained", True),
            token_stage=cfg["model"]["spatial"].get("token_stage", 3),
            use_npr_channel=cfg["model"]["spatial"].get("use_npr_channel", False),
            image_size=cfg["data"]["image_size"],
        )
        return SpatialClassifier(s, cfg["model"].get("head", {}).get("dropout", 0.1))
    if stage == "frequency":
        fq = cfg["model"]["frequency"]
        f = FrequencyStream(
            n_bands=fq.get("n_bands", 12), drop_lowest=fq.get("drop_lowest", 1),
            representation=fq.get("representation", "magnitude"),
            embed_dim=fq.get("embed_dim", 160), depth=fq.get("depth", 2),
            heads=fq.get("heads", 4), use_radial_mlp=fq.get("use_radial_mlp", True),
            image_size=cfg["data"]["image_size"],
        )
        return FrequencyClassifier(f, cfg["model"].get("head", {}).get("dropout", 0.1))
    return LEDDDetector(cfg)


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    tc = cfg["train"]
    if hasattr(model, "param_groups"):
        groups = model.param_groups(tc["lr_backbone"], tc["lr_new"], tc.get("weight_decay", 0.05))
    else:
        backbone = [p for n, p in model.named_parameters() if n.startswith("stream.backbone") and p.requires_grad]
        rest = [p for n, p in model.named_parameters() if not n.startswith("stream.backbone") and p.requires_grad]
        groups = [
            {"params": backbone, "lr": tc["lr_backbone"], "weight_decay": tc.get("weight_decay", 0.05)},
            {"params": rest, "lr": tc["lr_new"], "weight_decay": tc.get("weight_decay", 0.05)},
        ]
    return torch.optim.AdamW(groups)


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate_loader(model, loader, device, amp: bool = True) -> Dict[str, float]:
    model.eval()
    logits, labels = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], enabled=amp and device != "cpu"):
            out = model(x)
        logits.append(out["logit"].float().cpu())
        labels.append(batch["label"])
    if not logits:
        return {"auc": float("nan"), "n": 0}
    return classification_metrics(torch.cat(logits).numpy(), torch.cat(labels).numpy())


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    stage = cfg.get("stage", "joint")
    tc, dc = cfg["train"], cfg["data"]
    ckpt_dir = tc["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    log = get_logger("ledd", os.path.join(ckpt_dir, "train.log"))
    csv = CSVLogger(os.path.join(ckpt_dir, "metrics.csv"))
    set_seed(cfg.get("seed", 42))

    device = cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA unavailable -- falling back to CPU")
        device = "cpu"

    # ---------------------------------------------------------------- data
    splits_path = os.path.join(ckpt_dir, "splits.json")
    if os.path.exists(splits_path):
        splits = load_splits(splits_path)
    else:
        spec = SplitSpec(
            train_generators=dc["train_generators"],
            val_generator=dc["val_generator"],
            test_generators=dc["test_generators"],
            seed=cfg.get("seed", 42),
        )
        splits = make_splits(dc["root"], spec)
        save_splits(splits, splits_path)
    log.info({k: len(v) for k, v in splits.items()})

    train_loader = build_loader(
        splits["train"], tc["batch_size"], train=True, augment_cfg=cfg.get("augment"),
        num_workers=dc.get("num_workers", 4), balanced=True, seed=cfg.get("seed", 42),
    )
    # Selection signal is the VALIDATION GENERATOR, never in-distribution val.
    val_loader = build_loader(
        splits["val_generator"], cfg["eval"]["batch_size"], train=False,
        num_workers=dc.get("num_workers", 4),
    )
    indist_loader = build_loader(
        splits["val_indist"], cfg["eval"]["batch_size"], train=False,
        num_workers=dc.get("num_workers", 4),
    )

    # --------------------------------------------------------------- model
    model = build_model(cfg).to(device)
    if stage == "joint" and cfg.get("init"):
        rep = model.load_stream_checkpoints(
            cfg["init"].get("spatial_ckpt"), cfg["init"].get("frequency_ckpt")
        )
        log.info(f"loaded stream checkpoints: { {k: len(v['missing']) for k, v in rep.items()} } missing keys")
        if cfg["init"].get("freeze_streams", False):
            model.freeze_streams(True)
            log.info("streams frozen (ablation)")

    criterion = CombinedLoss(cfg).to(device)
    # Band attention is only materialised when something actually consumes it.
    need_attn = cfg["loss"].get("band_entropy_weight", 0.0) > 0 and stage in ("joint", "frequency")
    set_attention_recording(model, store_attn=need_attn, store_grad=False)

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * tc["epochs"]
    scheduler = cosine_with_warmup(optimizer, int(tc.get("warmup_epochs", 2)) * steps_per_epoch, total_steps)
    use_amp = cfg.get("amp", True) and device != "cpu"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    start_epoch, global_step, best = 0, 0, -1.0
    resume_path = find_resume(ckpt_dir, tc.get("resume", "auto"))
    if resume_path:
        st = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch, global_step, best = st["epoch"] + 1, st["global_step"], st["best_metric"]
        log.info(f"resumed from {resume_path} @ epoch {start_epoch} (best={best:.4f})")

    # ---------------------------------------------------------------- loop
    for epoch in range(start_epoch, tc["epochs"]):
        model.train()
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        t0, running = time.time(), {}
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.split(":")[0], enabled=use_amp):
                out = model(x, return_embeddings=True) if stage == "joint" else model(x)
                losses = criterion(out, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            if tc.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            for k, v in losses.items():
                if k != "loss":
                    running[k] = running.get(k, 0.0) + float(v)
            if global_step % tc.get("log_interval", 50) == 0:
                pbar.set_postfix({k: f"{v/max(global_step%steps_per_epoch,1):.3f}"
                                  for k, v in list(running.items())[:3]})

        val = evaluate_loader(model, val_loader, device, use_amp)
        indist = evaluate_loader(model, indist_loader, device, use_amp)
        metric = val.get("auc", float("nan"))
        is_best = not math.isnan(metric) and metric > best
        best = max(best, metric if not math.isnan(metric) else best)

        row = {
            "epoch": epoch, "step": global_step, "secs": round(time.time() - t0, 1),
            **{f"train_{k}": round(v / steps_per_epoch, 4) for k, v in running.items()},
            "val_gen_auc": round(val.get("auc", float("nan")), 4),
            "val_gen_f1": round(val.get("f1", float("nan")), 4),
            "indist_auc": round(indist.get("auc", float("nan")), 4),
            "best": round(best, 4),
        }
        csv.log(row)
        log.info(row)

        save_checkpoint(
            os.path.join(ckpt_dir, "last.pth"), model, optimizer, scheduler, scaler,
            epoch=epoch, global_step=global_step, best_metric=best,
            extra={"cfg": cfg}, is_best=is_best,
        )

    return {"best_val_generator_auc": best, "ckpt_dir": ckpt_dir}
