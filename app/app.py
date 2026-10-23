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
import json
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


def load_bundle(path: str, device: str = "cpu", max_tiles: int = 16,
                threshold: float | None = None):
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
    # The bundle's threshold (-7.52) maximises F1 on a 50/50 ADM split of 224-crop PNGs.
    # On arbitrary uploads -- full-resolution JPEGs, different crop positions, platform
    # resampling -- real photographs shift several logits toward "fake" and that cutoff
    # flags 53% of them. Measured on 66 real photographs and 120 full-resolution SDXL
    # generations, both tiled: -3.0 gives 9% false positives at 90% detection, -2.0 gives
    # 4.5% at 77.5%. Same weights, same ranking, different point on the ROC; AUC is
    # unaffected. The paper's number stays in the bundle, overrides are explicit.
    STATE.update(
        model=model, device=device,
        threshold=float(b["threshold"] if threshold is None else threshold),
        paper_threshold=float(b["threshold"]),
        n_bands=int(b["meta"]["n_bands"]), size=int(b["meta"]["image_size"]),
        meta=b["meta"], max_tiles=max_tiles,
    )
    return STATE


def load_references(path):
    """Measured score distributions the uploaded image is positioned against.

    Without these the app can only compare a score to a threshold, and on arbitrary
    uploads a threshold is not defensible: real photographs and some generators
    overlap, so no cut separates them. Showing where the score falls among measured
    populations reports the evidence instead of inventing a verdict.
    """
    if path and os.path.exists(path):
        STATE["references"] = json.load(open(path))
        return True
    return False


def pct_rank(x: float, pct: dict) -> float:
    """Percentile of x within a population summarised by a {quantile: value} dict."""
    qs = sorted(int(q) for q in pct)
    vs = [pct[str(q)] for q in qs]
    return float(np.clip(np.interp(x, vs, qs), 0, 100))


def scale_figure(logit: float) -> Image.Image:
    """Where this score sits among measured populations."""
    refs = STATE.get("references")
    pops = list(refs["populations"].items()) if refs else []
    fig, ax = plt.subplots(figsize=(9, 1.05 + 0.62 * max(1, len(pops))))
    colours = ["#2E5395", "#C55A11", "#2C6E49"]

    lo = min([logit] + [p["pct"]["5"] for _, p in pops]) - 1.2
    hi = max([logit] + [p["pct"]["95"] for _, p in pops]) + 1.2

    # overlap band: where the highest-reaching real population and the lowest-reaching
    # generated population both have mass. Drawn, not hidden -- it is the honest part.
    if len(pops) >= 2:
        a = max(p["pct"]["5"] for k, p in pops if "gener" in k.lower())
        b = min(p["pct"]["95"] for k, p in pops if "real" in k.lower())
        if a < b:
            ax.axvspan(a, b, color="#888", alpha=0.18, zorder=0)
            ax.text((a + b) / 2, len(pops) - 0.38, "overlap", ha="center", fontsize=7.5,
                    color="#555", style="italic")

    for i, (name, pop) in enumerate(pops):
        y = len(pops) - 1 - i
        c = colours[i % len(colours)]
        q = pop["pct"]
        ax.plot([q["5"], q["95"]], [y, y], color=c, lw=1.6, solid_capstyle="butt", zorder=2)
        ax.add_patch(plt.Rectangle((q["25"], y - 0.17), q["75"] - q["25"], 0.34,
                                   facecolor=c, alpha=0.38, edgecolor=c, zorder=3))
        ax.plot([q["50"]], [y], marker="|", ms=16, color=c, mew=2.2, zorder=4)
        ax.text(lo + 0.1, y + 0.3, f"{name}  (n={pop['n']})", fontsize=8.5, color=c)

    tp = (refs or {}).get("threshold_paper")
    if tp is not None and lo < tp < hi:
        ax.axvline(tp, color="#999", ls=":", lw=1.2)
        ax.text(tp, -0.72, "paper\nthreshold", ha="center", fontsize=7, color="#777")

    ax.axvline(logit, color="#c0392b", lw=2.4, zorder=6)
    ax.text(logit, len(pops) - 0.05, f"  this image {logit:+.2f}", color="#c0392b",
            fontsize=9.5, fontweight="bold", va="bottom")

    ax.set_xlim(lo, hi); ax.set_ylim(-0.95, len(pops) + 0.15)
    ax.set_yticks([]); ax.set_xlabel("model score (logit) - higher means more generator-like",
                                     fontsize=9)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120); plt.close(fig)
    return Image.open(buf)


def grid_overlay(img: Image.Image, crops_n: int, size: int, boxes) -> Image.Image:
    """The sampling grid drawn on the original.

    A contact sheet of the crops looks like a scrambled collage, because the crops are
    a sampling grid WITH GAPS (at 1280px wide the x-positions are 0/352/704/1056, each
    224 wide) rather than a contiguous mosaic. Pasting disjoint samples edge to edge
    invites the reader to think the tiling is broken. Showing the boxes in place does not.
    """
    from PIL import ImageDraw

    w, h = img.size
    scale = min(1.0, 760 / max(w, h))
    view = img.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                     Image.LANCZOS)
    d = ImageDraw.Draw(view, "RGBA")
    for (x, y) in boxes:
        d.rectangle([x * scale, y * scale, (x + size) * scale, (y + size) * scale],
                    outline=(255, 140, 0, 255), width=2, fill=(255, 140, 0, 28))
    return view


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
        return [crop], up, [(0, 0)]

    # Enough positions per axis that consecutive crops touch or overlap -- a grid with
    # gaps skips image, and nothing says the skipped strips carry less evidence. Then
    # thin out only if the count would exceed max_tiles.
    def positions(extent):
        k = math.ceil((extent - size) / size) + 1          # stride <= size => no gaps
        return [0] if k <= 1 else [round(i * (extent - size) / (k - 1)) for i in range(k)]

    xs, ys = positions(w), positions(h)
    while len(xs) * len(ys) > max_tiles and (len(xs) > 1 or len(ys) > 1):
        if len(xs) >= len(ys) and len(xs) > 1:
            xs = xs[::2] if len(xs) > 2 else xs[:1]
        elif len(ys) > 1:
            ys = ys[::2] if len(ys) > 2 else ys[:1]
    boxes = [(x, y) for y in ys for x in xs]
    return [img.crop((x, y, x + size, y + size)) for (x, y) in boxes], False, boxes


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
        return None, "Upload an image first.", "", None, None, None
    model, dev = STATE["model"], STATE["device"]
    n, T = STATE["n_bands"], STATE["threshold"]

    crops, upscaled, boxes = tile(img, STATE["size"], STATE["max_tiles"])
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

    refs = STATE.get("references")
    ranks = {k: pct_rank(logit, v["pct"]) for k, v in refs["populations"].items()} if refs else {}
    if ranks:
        # The overlap band is where both measured populations have mass. Inside it the
        # model genuinely cannot separate the two, and saying so is the correct output --
        # a confident answer there would be invented, not measured.
        gen = [v["pct"] for k, v in refs["populations"].items() if "gener" in k.lower()]
        rel = [v["pct"] for k, v in refs["populations"].items() if "real" in k.lower()]
        lo = max(q["5"] for q in gen) if gen else T
        hi = min(q["95"] for q in rel) if rel else T
        if logit < min(lo, hi):
            headline = "PROBABLY A REAL PHOTO"
            plain = ("It scores below the range where generated images start to appear. "
                     f"Of the {refs['populations']['real photographs']['n']} real photographs "
                     f"we measured, about {100 - ranks['real photographs']:.0f}% scored "
                     "higher than this one.")
        elif logit > max(lo, hi):
            headline = "PROBABLY AI-GENERATED"
            plain = ("It scores above almost every real photograph we measured, in the "
                     "range typical of an image generator.")
        else:
            headline = "TOO CLOSE TO CALL"
            plain = ("Its score lands in the band where real photographs and generated "
                     "images overlap. This model cannot separate the two here, and a "
                     "confident answer would be guesswork.")
        plain += (" This is a leaning, not a verdict - the model was trained on one "
                  "benchmark and image generators it has never seen may score anywhere.")
    else:
        headline = "LIKELY GENERATED" if is_fake else "LIKELY REAL"
        plain = ""
    top = int(np.argmax(drops))
    span = float(np.max(np.abs(drops)))
    iw, ih = img.size
    covered = set()
    for bx, by in boxes:                       # union, so overlap is not double-counted
        covered |= {(a_, b_) for a_ in range(bx, bx + STATE["size"], 32)
                    for b_ in range(by, by + STATE["size"], 32)}
    coverage = min(100.0, 100.0 * len(covered) /
                   max(1, len(range(0, iw, 32)) * len(range(0, ih, 32))))
    if coverage < 60:
        plain += (f" Note: this image is large, so only about {coverage:.0f}% of it was "
                  "sampled - the rest was not looked at.")
    md = (
        f"### {headline}\n\n"
        f"{plain}\n\n"
        f"- Score **{score:.3f}** (0.5 = the decision threshold). Mean logit {logit:+.3f} "
        f"against threshold {T:+.3f}; raw sigmoid would read {p_raw:.4f}, which is why this "
        f"app does not show it.\n"
        f"- Read from **{len(crops)} crop{'s' if len(crops) != 1 else ''}** of "
        f"{STATE['size']}px at native scale covering **{coverage:.0f}%** of the image, "
        f"per-crop logits {per_crop.min():+.2f} to {per_crop.max():+.2f} "
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
    lead, _, detail = md.partition("- Score ")
    detail = "- Score " + detail
    return (scale_figure(logit), lead, detail, chart, rows,
            grid_overlay(img, len(crops), STATE["size"], boxes))


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
                crop_view = gr.Image(label="Sampling grid - the crops read, in place",
                                     height=300)
            with gr.Column(scale=2):
                if abs(STATE["threshold"] - STATE["paper_threshold"]) > 1e-9:
                    gr.Markdown(
                        f"*Threshold {STATE['threshold']:+.2f}, overriding the paper's "
                        f"{STATE['paper_threshold']:+.2f}. The paper value maximises F1 on a "
                        f"balanced benchmark split and flags roughly half of ordinary "
                        f"photographs; this one trades detection for far fewer false alarms. "
                        f"Table I's figures use the paper value.*")
                verdict = gr.Markdown()
                scale = gr.Image(label="Where this score sits", height=190)
                with gr.Accordion("Technical detail", open=False):
                    detail = gr.Markdown()
                    chart = gr.Image(label="Band attribution", height=420)
                    table = gr.Dataframe(
                        headers=["band (radius)", "causal drop", "attention"],
                        label="Per-band detail", wrap=True,
                    )
        outs = [scale, verdict, detail, chart, table, crop_view]
        btn.click(analyse, inputs=inp, outputs=outs)
        inp.change(analyse, inputs=inp, outputs=outs)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="runs/export/ledd_detector.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--max-tiles", type=int, default=36,
                    help="cap on crops per image; the grid is contiguous, so this only "
                         "bites on very large images")
    ap.add_argument("--references", default=None,
                    help="references.json of measured score populations; without it the "
                         "app falls back to a bare threshold verdict")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the bundle's F1-calibrated threshold. -3.0 is a "
                         "measured demo operating point: 9%% false positives on real "
                         "photographs at 90%% detection on SDXL. The bundle default "
                         "(-7.52) flags 53%% of real photographs.")
    a = ap.parse_args()

    load_bundle(a.bundle, a.device, a.max_tiles, a.threshold)
    ref_path = a.references or os.path.join(os.path.dirname(os.path.abspath(a.bundle)),
                                            "references.json")
    print("references:", "loaded" if load_references(ref_path) else f"NOT FOUND at {ref_path}")
    print(f"loaded {a.bundle} on {a.device}: "
          f"{STATE['meta']['n_params_deployed']:,} params, threshold {STATE['threshold']:+.4f}"
          + ("" if a.threshold is None
             else f" (OVERRIDE; paper value {STATE['paper_threshold']:+.4f})"))
    build_ui().launch(server_port=a.port, share=a.share)
