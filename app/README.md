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
| Band attention | displayed **for contrast**, labelled as not faithful |
| Spatial heatmap | deliberately absent |

The two attribution panels usually disagree, and that disagreement is the
paper's Section V-E: attention concentrates mid-band, while deleting the
highest band is what actually costs the model its decision.

The spatial relevance maps pass the insertion test (0.765 against a 0.566 random
control) but fail deletion (0.625 against 0.577). A heatmap in a user interface
reads as authoritative whether or not it has earned it, so it is left out until
the deletion confound (constant-fill masking creates unnaturally smooth patches
that a synthetic-image detector reacts to) is resolved.

## Limits worth repeating to anyone who uses this

- Trained on GenImage; calibrated on one held-out generator. Transfer to an
  unseen generator is measured (0.975 AUC on Midjourney, 0.983 on SDXL) but not
  guaranteed for arbitrary pipelines.
- Photographic content only. Digital art, renders and fractal imagery have
  different spectral statistics and are out of scope.
- Answers *was this generated*, not *which region was edited*.
- Images below the crop size are upscaled, which imprints a resampling
  signature. The app flags this when it happens.
