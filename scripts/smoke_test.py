#!/usr/bin/env python
"""End-to-end smoke test. Run this FIRST on any new machine, before any real data.

Exercises exactly the paths that have never executed and are most likely to break:
MobileViT token-stage indexing against the real timm feature list, autocast +
recorded-attention interaction, Chefer relevance shapes through the fusion block,
band masking, and the loss/backward path.

    python scripts/smoke_test.py            # CPU, ~1 minute
    python scripts/smoke_test.py --cuda     # also checks AMP on GPU
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = []


def check(name):
    def deco(fn):
        def wrapped(*a, **k):
            try:
                detail = fn(*a, **k) or ""
                RESULTS.append((True, name, detail))
            except Exception as e:
                RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
                if os.environ.get("SMOKE_VERBOSE"):
                    traceback.print_exc()
        return wrapped
    return deco


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--full-size", action="store_true",
                    help="build at 224 with pretrained weights (downloads ~22MB)")
    args = ap.parse_args()

    import torch

    from ledd.explain import (deletion_insertion_bands, deletion_insertion_pixels,
                              explain_batch, normalize_map, stream_deletion_effect)
    from ledd.losses import CombinedLoss
    from ledd.models import LEDDDetector, count_parameters
    from ledd.models.attention import set_attention_recording

    size = 224 if args.full_size else 128
    cfg = {
        "seed": 0,
        "data": {"image_size": size},
        "model": {
            "spatial": {"name": "mobilevit_s", "pretrained": args.full_size},
            "frequency": {"n_bands": 12, "drop_lowest": 1, "representation": "magnitude",
                          "embed_dim": 160, "depth": 2, "heads": 4, "use_radial_mlp": True},
            "fusion": {"mode": "cross_attention", "dim": 192, "heads": 4, "layers": 1,
                       "bidirectional": True, "modality_dropout": 0.25},
            "head": {"hidden": 192, "dropout": 0.1},
            "projection": {"dim": 128, "hidden": 192},
        },
        "loss": {
            "bce_weight": 1.0, "supcon_weight": 0.5, "band_entropy_weight": 0.05,
            "supcon": {"temperature": 0.1, "queue_size": 256,
                       "warmup_steps": 0, "ramp_steps": 1},
            "band_entropy": {"normalize": True},
        },
    }

    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"device={device}  image_size={size}  torch={torch.__version__}")

    state = {}

    @check("model builds")
    def _build():
        state["model"] = LEDDDetector(cfg).to(device)
        red = state["model"].spatial.reductions
        stage = state["model"].spatial.token_stage
        return f"token stage {stage} (stride {red[stage]}), grid {state['model'].spatial.grid_size}"
    _build()
    if "model" not in state:
        report(); return 1
    model = state["model"]

    @check("parameter count is lightweight")
    def _params():
        n = count_parameters(model)
        assert n < 9_000_000, f"{n:,} params -- too big"
        return f"{n:,} total"
    _params()

    x = torch.rand(4, 3, size, size, device=device)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)

    @check("forward pass + explainability outputs")
    def _fwd():
        set_attention_recording(model, True)
        out = model(x, return_embeddings=True)
        assert out["logit"].shape == (4,), out["logit"].shape
        assert out["band_attention"].shape == (4, 12)
        assert out["balance"].shape == (4, 2)
        assert torch.allclose(out["band_attention"].sum(1), torch.ones(4, device=device), atol=1e-4)
        state["out"] = out
        return f"logit{tuple(out['logit'].shape)} bands{tuple(out['band_attention'].shape)}"
    _fwd()

    @check("combined loss + backward")
    def _loss():
        crit = CombinedLoss(cfg).to(device)
        losses = crit(state["out"], y)
        losses["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, "no gradients"
        assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
        return " ".join(f"{k}={float(v):.3f}" for k, v in losses.items() if k != "loss")
    _loss()

    @check("optimizer step changes weights")
    def _step():
        before = model.head[-1].weight.detach().clone()
        opt = torch.optim.AdamW(model.param_groups(1e-4, 1e-3))
        opt.step()
        assert not torch.allclose(before, model.head[-1].weight), "weights unchanged"
    _step()

    if device == "cuda":
        @check("AMP autocast + GradScaler")
        def _amp():
            scaler = torch.amp.GradScaler()
            opt = torch.optim.AdamW(model.param_groups(1e-5, 1e-3))
            with torch.autocast("cuda"):
                out = model(x, return_embeddings=True)
                loss = CombinedLoss(cfg).to(device)(out, y)["loss"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            assert torch.isfinite(loss)
            return f"loss={float(loss):.3f}"
        _amp()

    model.eval()

    @check("band + spatial masking changes prediction")
    def _mask():
        base = model(x)["logit"]
        nb = model(x, band_mask=torch.zeros(4, 12, device=device))["logit"]
        gh, gw = model(x)["grid"]
        ns = model(x, spatial_token_mask=torch.zeros(4, gh * gw, device=device))["logit"]
        assert not torch.allclose(base, nb, atol=1e-4), "band mask had no effect"
        assert not torch.allclose(base, ns, atol=1e-4), "spatial mask had no effect"
        return f"dband={float((base-nb).abs().mean()):.3f} dspatial={float((base-ns).abs().mean()):.3f}"
    _mask()

    @check("Chefer relevance maps (the untested path)")
    def _explain():
        r = explain_batch(model, x[:2])
        assert r["spatial_map"].shape == (2, size, size), r["spatial_map"].shape
        assert r["band_attribution"].shape == (2, 12)
        assert r["band_to_region"].shape == (2, 12, size, size)
        state["sal"] = normalize_map(r["spatial_map"])
        state["attr"] = r["band_attribution"]
        return "spatial_map, band_attribution, band_to_region all correct shape"
    _explain()

    @check("deletion / insertion AUC")
    def _faith():
        if "sal" not in state:
            raise RuntimeError("skipped: explain_batch failed")
        p = deletion_insertion_pixels(model, x[:2], state["sal"], steps=6)
        b = deletion_insertion_bands(model, x[:2], state["attr"])
        for k in ("deletion_auc", "insertion_auc"):
            assert 0 <= p[k] <= 1, f"{k}={p[k]}"
        return f"pixel del={p['deletion_auc']:.3f} ins={p['insertion_auc']:.3f} | band del={b['band_deletion_auc']:.3f}"
    _faith()

    @check("stream-deletion causal balance")
    def _bal():
        r = stream_deletion_effect(model, x[:2])
        assert r["causal_balance"].shape == (2, 2)
        return f"mean frequency share={float(r['causal_balance'][:,1].mean()):.3f}"
    _bal()

    @check("attention recording disabled afterwards")
    def _rec():
        from ledd.models.attention import RecordedAttention
        assert all(not m.store_attn for m in model.modules() if isinstance(m, RecordedAttention))
    _rec()

    return report()


def report() -> int:
    print()
    width = max(len(n) for _, n, _ in RESULTS) + 2
    for ok, name, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}} {detail}")
    n_ok = sum(ok for ok, _, _ in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    if n_ok < len(RESULTS):
        print("Re-run with SMOKE_VERBOSE=1 for tracebacks.")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
