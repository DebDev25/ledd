#!/usr/bin/env python
"""LEDD demo -- upload an image, get a calibrated probability and a CAUSAL
frequency-band attribution.

A note on what this shows, and what it deliberately does not
------------------------------------------------------------
The obvious thing to display would be the frequency stream's attention over
bands. We do not make that the headline, because our own experiments showed it
is not an explanation: attention concentrates mid-band while *deleting* the
highest band costs 0.270 AUC and deleting the mid-band rings costs roughly
nothing. Shipping attention as the explanation would be shipping a number we
have already demonstrated to be misleading.

What the app shows instead is the causal effect: each band is zeroed in turn and
we measure how far the model's confidence in its own decision falls. That is
12 extra forward passes -- cheap, exact (a ring is zeroed, not approximated),
and it is the quantity the paper's conclusions rest on.

Attention is still displayed, clearly labelled, side by side. The disagreement
between the two panels is the interesting part.

Spatial heatmaps are not included. The relevance maps pass insertion (0.765 vs
0.566 random) but not deletion (0.625 vs 0.577), so they are suggestive rather
than trustworthy, and a heatmap in a UI reads as authoritative whether or not it
has earned it.

Run:
    pip install -r app/requirements.txt
    python app/app.py --bundle runs/export/ledd_detector.pt
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys

import gradio as gr
import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ledd.data.dataset import to_tensor_normalized  # noqa: E402
from ledd.data.prepare import center_crop_or_resize  # noqa: E402
from ledd.engine.train import build_model  # noqa: E402
from ledd.models.attention import set_attention_recording  # noqa: E402

STATE = {}


def load_bundle(path: str, device: str = "cpu", max_tiles: int = 16):
    b = torch.load(path, map_location="cpu", weights_only=False)
    if b.get("format") != "ledd-bundle-v1":
        raise ValueError(f"{path} is not a LEDD export bundle; run scripts/export_model.py")
    model = build_model(b["config"])
    model.load_state_dict(b["state_dict"], strict=False)   # projection head is absent by design
    model.eval().to(device)
    # band_attention() returns None unless recording is on -- it reads the stored
    # attention tensor of the last block. Without this the attention panel would be
    # silently all-zero, which on this of all projects would be the wrong bug to ship.
    set_attention_recording(model, store_attn=True, store_grad=False)
    STATE.update(
        model=model, device=device, threshold=float(b["threshold"]),
        n_bands=int(b["meta"]["n_bands"]), size=int(b["meta"]["image_size"]),
        meta=b["meta"], max_tiles=max_tiles,
    )
    return STATE


def band_labels(n: int):
    return [f"{math.sqrt(k / n):.2f}-{math.sqrt((k + 1) / n):.2f}" for k in range(n)]


def tile(img: Image.Image, size: int, max_tiles: int = 16):
    """Cover the whole image with `size` x `size` crops -- no resizing.

    A single centre crop sees `size`^2 pixels of what may be a 1024x1536 image, so the
    verdict depends on where the crop happened to land. Downscaling the image first
    would use all of it, but resampling imprints the kernel's own spectral signature,
    which is exactly the artifact class this detector reads -- so that cure is worse
    than the disease. Tiling keeps every pixel at native scale and aggregates.

    Returns (list[Image], upscaled_flag). Falls back to one centre crop when the image
    is too small to tile.
    """
    img = img.convert("RGB")
    w, h = img.size
    if w < size or h < size:
        crop, up = center_crop_or_resize(img, size)
        return [crop], up

    per_axis = max(1, int(math.isqrt(max_tiles)))
    nx = min(per_axis, max(1, w // size))
    ny = min(per_axis, max(1, h // size))
    xs = [0] if nx == 1 else [round(i * (w - size) / (nx - 1)) for i in range(nx)]
    ys = [0] if ny == 1 else [round(i * (h - size) / (ny - 1)) for i in range(ny)]
    return [img.crop((x, y, x + size, y + size)) for y in ys for x in xs], False


def montage(crops, cell: int = 96, cols: int = 4) -> Image.Image:
    """Contact sheet of what the model actually saw."""
    cols = min(cols, len(crops))
    rows = math.ceil(len(crops) / cols)
    sheet = Image.new("RGB", (cols * cell, rows * cell), "#1a1a1a")
    for i, c in enumerate(crops):
        sheet.paste(c.resize((cell, cell), Image.LANCZOS), ((i % cols) * cell, (i // cols) * cell))
    return sheet


@torch.no_grad()
def analyse(img: Image.Image):
    if img is None:
        return None, "Upload an image first.", None, None
    model, dev = STATE["model"], STATE["device"]
    n, T = STATE["n_bands"], STATE["threshold"]

    crops, upscaled = tile(img, STATE["size"], STATE["max_tiles"])
    x = torch.stack([to_tensor_normalized(c) for c in crops]).to(dev)

    out = model(x)
    per_crop = out["logit"].float().cpu().numpy()
    logit = float(per_crop.mean())
    is_fake = logit >= T

    # Score against the ACTUAL decision boundary, not logit 0. sigmoid(logit) answers
    # "how far above zero?", but this model's operating point is T = -7.52, where
    # sigmoid(T) is 0.0005 -- so a correctly-detected fake shows P = 0.002 and looks
    # like a confident REAL. Shifting by T puts 0.5 at the threshold, which is what a
    # reader assumes a score means. This is a monotone rescaling for display, not a
    # Platt-calibrated probability; the raw logit is still shown.
    score = float(1.0 / (1.0 + math.exp(-(logit - T))))
    p_raw = float(1.0 / (1.0 + math.exp(-logit)))

    # ---- causal band attribution: zero one band at a time
    #
    # Measured in LOGIT space, not probability. On a confident image the sigmoid is
    # saturated -- P(real) sits at ~1.0, so deleting a band moves it by less than float
    # precision and every bar underflows to exactly 0.0. The logit has no ceiling, so it
    # still shows which bands carry evidence even when the decision is not in doubt.
    # Sign is oriented toward the predicted class, so positive always means
    # "deleting this band cost the model evidence for what it decided".
    sign = 1.0 if is_fake else -1.0
    C = x.shape[0]
    drops = []
    for k in range(n):
        m = torch.ones(C, n, device=dev)
        m[:, k] = 0.0
        lk = float(model(x, band_mask=m)["logit"].float().mean().item())
        drops.append(sign * (logit - lk))

    # ---- attention, for contrast only
    raw_att = out.get("band_attention")
    have_att = raw_att is not None
    att = raw_att.float().mean(0).cpu().numpy() if have_att else np.zeros(n)

    labels = band_labels(n)
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    ax[0].bar(labels, drops, color=["#c0392b" if d > 0 else "#7f8c8d" for d in drops])
    ax[0].set_title("Causal: evidence lost when this band is deleted  (this is the explanation)",
                    fontsize=10, loc="left")
    ax[0].set_ylabel("logit drop")
    ax[0].axhline(0, lw=0.8, color="k")
    ax[1].bar(labels, att, color="#34495e")
    ax[1].set_title(
        "Attention over bands  (not faithful IN AGGREGATE - may coincide on any one image)"
        if have_att
        else "Attention unavailable - recording is off, so this panel is empty",
        fontsize=10, loc="left")
    ax[1].set_ylabel("attention")
    ax[1].set_xlabel("frequency band (normalised radius)")
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    chart = Image.open(buf)

    verdict = "LIKELY GENERATED" if is_fake else "LIKELY REAL"
    top = int(np.argmax(drops))
    span = float(np.max(np.abs(drops)))
    md = (
        f"### {verdict}\n\n"
        f"- Score **{score:.3f}** (0.5 = the decision threshold). Mean logit {logit:+.3f} "
        f"against threshold {T:+.3f}; raw sigmoid would read {p_raw:.4f}, which is why this "
        f"app does not show it.\n"
        f"- Read from **{len(crops)} crop{'s' if len(crops) != 1 else ''}** of "
        f"{STATE['size']}px at native scale, per-crop logits "
        f"{per_crop.min():+.2f} to {per_crop.max():+.2f} "
        f"({int((per_crop >= T).sum())}/{len(crops)} above threshold)\n"
        f"- Largest causal band: **{labels[top]}**, worth {drops[top]:+.3f} logits "
        f"out of {abs(logit - T):.2f} above threshold\n"
        f"- Band 0 always reads ~0: the DC component and lowest ring are zeroed "
        f"inside the model, since they carry image content rather than generator artifacts.\n"
    )
    if span < 0.05:
        md += (
            "\n> **No frequency band meaningfully changes this decision** "
            f"(largest effect {span:.3f} logits). That is not a bug and not a shrug: it is "
            "this paper's result. In the reported concatenation model the frequency stream "
            "contributes 0.003 AUC at inference, and no band costs more than 0.003. The "
            "stream earns its place during *training* (+0.0146 AUC, 3.7 sigma over three "
            "seeds), not at prediction time. The band attribution that does show structure — "
            "band 11 worth 0.270 AUC — belongs to the standalone frequency stream, a "
            "different model.\n"
        )
    if upscaled:
        md += ("\n> This image was smaller than "
               f"{STATE['size']}px and had to be upscaled. Resampling imprints its own "
               "spectral signature, so treat the frequency attribution with caution.\n")
    md += (
        "\n---\n**Read this before trusting the number.** The threshold was fitted on one "
        "held-out generator (ADM) and the model was trained on GenImage. On a generator or "
        "an editing pipeline unlike anything in that benchmark, calibration is not "
        "guaranteed to hold. Treat the output as evidence, not proof - and note the model "
        "answers *was this generated*, not *which part was edited*."
    )

    if len(crops) > 1 and 0 < int((per_crop >= T).sum()) < len(crops):
        md += ("\n> **The crops disagree with each other.** Some regions read generated and "
               "others real. Flat sky or blur carries little generator signal either way, so "
               "a split vote often means part of the frame is uninformative rather than that "
               "part of the image is authentic. This model answers a whole-image question; "
               "it was never trained to localise.\n")
    rows = [[labels[k], round(drops[k], 4), round(float(att[k]), 4)] for k in range(n)]
    return chart, md, rows, montage(crops)


def build_ui():
    m = STATE["meta"]
    with gr.Blocks(title="LEDD - diffusion image detector") as demo:
        gr.Markdown(
            f"# LEDD\n"
            f"Lightweight explainable detector for diffusion-generated images - "
            f"{m['n_params_deployed']:,} parameters, no diffusion model at inference.\n\n"
            f"The image is covered with {m['image_size']}px crops at native scale and the "
            f"verdict averaged over them. It is never resized: resampling imprints the "
            f"kernel's own spectral signature, which is the artifact class this detector "
            f"reads."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Image")
                btn = gr.Button("Analyse", variant="primary")
                crop_view = gr.Image(label="What the model actually sees (every crop, native scale)",
                                     height=240)
            with gr.Column(scale=2):
                verdict = gr.Markdown()
                chart = gr.Image(label="Band attribution", height=420)
                table = gr.Dataframe(
                    headers=["band (radius)", "causal drop", "attention"],
                    label="Per-band detail", wrap=True,
                )
        btn.click(analyse, inputs=inp, outputs=[chart, verdict, table, crop_view])
        inp.change(analyse, inputs=inp, outputs=[chart, verdict, table, crop_view])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="runs/export/ledd_detector.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--max-tiles", type=int, default=16,
                    help="cap on crops per image; 16 is ~1s on CPU")
    a = ap.parse_args()

    load_bundle(a.bundle, a.device, a.max_tiles)
    print(f"loaded {a.bundle} on {a.device}: "
          f"{STATE['meta']['n_params_deployed']:,} params, threshold {STATE['threshold']:+.4f}")
    build_ui().launch(server_port=a.port, share=a.share)
