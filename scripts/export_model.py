#!/usr/bin/env python
"""Export a trained checkpoint as a self-contained deployment bundle.

Why not pickle the model object
-------------------------------
`torch.save(model)` pickles the *class*, not the weights -- it stores a reference
to `ledd.models.detector.LEDDDetector` and replays `__init__` on load. That bundle
breaks the moment this repo's module layout changes, breaks again on a different
torch/timm version, and cannot be loaded at all without the source tree present.
Worse, unpickling executes arbitrary code, so a bundle is only as trustworthy as
whoever sent it.

We save a *state_dict plus the config that built it*. `torch.save` still uses
pickle underneath, but only over tensors and plain dicts, so the file is inert
data: loadable with `weights_only=True`, portable across versions, and
reconstructible by `build_model(cfg)` without guessing hyperparameters.

The threshold is bundled too. Without it the model is unusable for a yes/no
answer: our logits are badly calibrated and the decision point was fitted on the
validation generator (ADM), never on the test generator.

    python scripts/export_model.py --config configs/stage3_joint.yaml \
        --ckpt runs/stage3_joint/best.pth \
        --calibration runs/stage3_joint/threshold.json \
        --out runs/export/ledd_detector.pt
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import torch
from _common import base_parser, get_config

from ledd.engine.train import build_model


def _git_sha() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    ap = base_parser(__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--calibration", default=None,
                    help="threshold.json from calibrate_threshold.py")
    ap.add_argument("--out", default="runs/export/ledd_detector.pt")
    ap.add_argument("--torchscript", action="store_true",
                    help="additionally attempt a TorchScript trace (may fail; see note)")
    args = ap.parse_args()
    cfg = get_config(args)

    model = build_model(cfg)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    model.load_state_dict(sd)
    model.eval()

    # The SupCon projection head is training-only. Dropping it keeps the
    # "6.3M parameters" claim honest for the thing we actually ship.
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("projection.")}

    # Count PARAMETERS, not state_dict entries. state_dict also carries buffers --
    # BatchNorm running_mean/running_var and friends, 12,288 of them here -- and summing
    # it reports 5,587,849 where the parameter count is 5,575,561. That 12k discrepancy
    # is small enough to look like a rounding artefact and is exactly the kind of number
    # that ends up in an abstract. Report the two separately.
    param_names = {n for n, _ in model.named_parameters()}
    n_params = sum(v.numel() for k, v in sd.items() if k in param_names)
    n_buffers = sum(v.numel() for k, v in sd.items() if k not in param_names)
    n_total = sum(p.numel() for p in model.parameters())

    threshold, calib = 0.0, None
    if args.calibration and os.path.exists(args.calibration):
        calib = json.load(open(args.calibration))
        threshold = float(calib.get("threshold", 0.0))

    bundle = {
        "format": "ledd-bundle-v1",
        "state_dict": sd,
        "config": cfg,
        "threshold": threshold,            # logit space, fitted on the val generator
        "calibration": calib,
        "meta": {
            "source_ckpt": os.path.basename(args.ckpt),
            "epoch": int(ck.get("epoch", -1)) if isinstance(ck, dict) else -1,
            "git_sha": _git_sha(),
            "torch": torch.__version__,
            "n_params_deployed": n_params,      # parameters only, projection head excluded
            "n_params_total": n_total,          # includes the training-only projection head
            "n_buffers": n_buffers,             # BatchNorm running stats etc., not parameters
            "fusion_mode": cfg["model"]["fusion"]["mode"],
            "image_size": cfg.get("data", {}).get("image_size", 224),
            "n_bands": cfg["model"]["frequency"].get("n_bands", 12),
            "preprocessing": "center_crop_no_resize; imagenet mean/std; FFT computed in-model",
            "threshold_fitted_on": (calib or {}).get("fit_on", "val_generator"),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(bundle, args.out)
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()[:16]
    print(json.dumps({"out": args.out, "sha256_16": sha,
                      "params_deployed": n_params, "params_total": n_total,
                      "buffers": n_buffers, "fusion_mode": bundle["meta"]["fusion_mode"],
                      "threshold": threshold,
                      "mb": round(os.path.getsize(args.out) / 1e6, 2)}, indent=2))

    if args.torchscript:
        # Honest caveat: the frequency stream calls torch.fft and builds ring masks
        # from tensor shapes. Tracing bakes in the input size and may silently drop
        # the band_mask path that the explanation code depends on. Reported, not hidden.
        try:
            ex = torch.randn(1, 3, bundle["meta"]["image_size"], bundle["meta"]["image_size"])
            ts = torch.jit.trace(model, ex, strict=False)
            p = args.out.replace(".pt", ".torchscript.pt")
            ts.save(p)
            print(f"torchscript: {p} (traced at fixed {ex.shape[-1]}px; "
                  "band_mask ablation is NOT available in the traced graph)")
        except Exception as e:  # noqa: BLE001
            print(f"torchscript: FAILED ({type(e).__name__}: {e}). "
                  "Use the state_dict bundle; the FFT path does not trace cleanly.")
