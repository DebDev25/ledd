# LEDD demo app

Upload an image, get a calibrated probability and a **causal** frequency-band
attribution.

## 1. Export a bundle (once, after training)

```bash
python scripts/export_model.py \
    --config configs/stage3_joint.yaml \
    --ckpt   runs/stage3_joint/best.pth \
    --calibration runs/stage3_joint/threshold.json \
    --out    runs/export/ledd_detector.pt
```

The bundle is a `state_dict` + the config that built it + the decision threshold.
It is **not** a pickled model object: see the docstring in `export_model.py` for
why that distinction matters for portability and for safety.

## 2. Run

```bash
pip install -r app/requirements.txt
python app/app.py --bundle runs/export/ledd_detector.pt          # http://127.0.0.1:7860
python app/app.py --bundle runs/export/ledd_detector.pt --share  # temporary public link
```

CPU is fine: 13 forward passes of a 6.3M-parameter model, roughly a second.

## What the app shows

| Panel | Status |
|---|---|
| P(generated) + calibrated verdict | threshold fitted on the validation generator (ADM) only |
| Causal band attribution | each band zeroed in turn; exact, not approximated |
| Band attention | displayed for contrast, labelled as not faithful *in aggregate* |
| Spatial heatmap | deliberately absent |

The two attribution panels usually disagree, and that disagreement is the
paper's Section V-E: attention concentrates mid-band, while deleting the
highest band is what actually costs the model its decision.

That evidence is dataset-level, measured on the standalone frequency stream
against the validation generator. On any single image the two panels may agree
by coincidence; "unfaithful" is a claim about the distribution, not a promise
about every example.

Band deletion is measured in **logit** space. In probability space a confident
prediction saturates the sigmoid and every bar underflows to exactly zero, which
reads as "no band matters" when the truth is "the decision was never in doubt".

The spatial relevance maps pass the insertion test (0.765 against a 0.566 random
control) but fail deletion (0.625 against 0.577). A heatmap in a user interface
reads as authoritative whether or not it has earned it, so it is left out until
the deletion confound (constant-fill masking creates unnaturally smooth patches
that a synthetic-image detector reacts to) is resolved.

## Choosing a threshold

The bundle carries **T = -7.52**, fitted to maximise F1 on a 50/50 ADM split of
224-crop PNGs. That is the right number for the paper and the wrong one for a demo.
On arbitrary uploads -- full-resolution JPEGs, arbitrary crop positions, platform
resampling -- real photographs shift several logits toward "fake", and that cutoff
flags **53% of them**.

Measured on 66 real photographs and 120 full-resolution SDXL generations, both tiled
the way the app tiles them:

| Threshold | False positives | Fakes caught |
|---|---|---|
| -7.52 (paper) | 53.0% | 100% |
| -3.00 | 9.1% | 90.0% |
| -2.00 | 4.5% | 77.5% |
| -1.00 | 1.5% | 53.3% |

```bash
python app/app.py --bundle runs/export/ledd_detector.pt --threshold -3.0
```

Same weights, same ranking, a different point on the ROC -- AUC is untouched, and the
paper's F1 figures still use -7.52. The app states loudly when it is overriding.

Two cautions. 66 reals makes the false-positive column coarse (9.1% is six images).
And both classes must be measured through the *same* pipeline: an earlier version of
this table scored reals as full-resolution JPEGs and fakes as 224-crop PNGs, which
understated detection at -3.00 by 18 points.

## Limits worth repeating to anyone who uses this

- Trained on GenImage; calibrated on one held-out generator. Transfer to an
  unseen generator is measured (0.975 AUC on Midjourney, 0.983 on SDXL) but not
  guaranteed for arbitrary pipelines.
- Photographic content only. Digital art, renders and fractal imagery have
  different spectral statistics and are out of scope.
- Answers *was this generated*, not *which region was edited*.
- Images below the crop size are upscaled, which imprints a resampling
  signature. The app flags this when it happens.
