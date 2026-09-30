# LEDD — Lightweight Explainable Diffusion Detection

Dual-stream detector for diffusion-generated images: MobileViT-S on RGB + a
radial band-token frequency stream, fused by bidirectional cross-attention with a
`[FUSE]` readout. ~7M parameters, intrinsic frequency-band attribution, causal
faithfulness evaluation.

See `../Architecture_Methodology_v2.docx` for the design rationale behind every
choice here.

## Status

Code complete and verified. All three training stages have been run.

| Result | Value |
|---|---|
| Model size | 6,306,633 parameters |
| Dataset | 160,000 images, 8 generators, leak check clean (0.130 vs 0.125) |
| Frequency stream alone | 0.876 in-dist → 0.829 unseen (gap 0.047) |
| Spatial stream alone | 0.9998 in-dist → 0.917 unseen (gap 0.083) |
| Joint, cross-attention | 0.946 best / 0.905 mean (swing 0.097) |
| Joint, concatenation | 0.937 best / **0.928 mean** (swing 0.018) |
| After threshold calibration | ADM F1 0.876 · Midjourney F1 0.897, AUC 0.963 |

**Two findings worth knowing before reading the code:**

1. **Simple concatenation beats cross-attention** on every stable measure. Reported as a
   negative result rather than selecting the noisier model's best epoch.
2. **Attention does not explain the decision.** Band attention peaks mid-band while causal
   deletion shows the highest band carries the signal (removing it: 0.830 → 0.560). At
   stream level, attention claims 97% frequency reliance where deletion shows 4%.

Remaining: seeds, ablations A3/A5/A6, the SDXL/SD3/Flux set, and baselines.

### Always calibrate before quoting accuracy or F1

`scripts/calibrate_threshold.py` fits the decision threshold on the **validation
generator** and applies it everywhere. AUC is threshold-free and unaffected, but F1 moved
from 0.775 to 0.876 with no change to the model.

**Nothing has been run on a GPU yet.** Config/data tests (13) pass; the
torch-dependent tests are written but need a machine with torch installed.

**Start here:** `notebooks/LEDD_kaggle_runbook.ipynb` — the working run book, with every
path absolute and each phase marked done or pending. `COLAB_SETUP.md` covers first-time
setup on a fresh machine.

## Install

```bash
pip install -r requirements.txt
```

## Order of operations

```bash
# 0. Build the crop archive (see ledd/data/prepare.py for crop-not-resize rationale)
python scripts/prepare_data.py --src /path/to/raw/GenImage --dst data/genimage_224 \
    --generators sd_v14 sd_v15 wukong vqdm biggan midjourney adm glide --n-per-class 10000

# 1. LEAK CHECK — before any GPU time. Exits non-zero if the pipeline leaks.
python scripts/run_leak_check.py --archive data/genimage_224 \
    --generators sd_v14 sd_v15 wukong vqdm biggan midjourney adm glide

# 2. Stages (each resumes automatically after a killed session)
python scripts/train.py --config configs/stage1_spatial.yaml
python scripts/train.py --config configs/stage2_frequency.yaml
python scripts/train.py --config configs/stage3_joint.yaml

# 3. Evaluation protocol + explainability + efficiency
python scripts/evaluate.py --config configs/stage3_joint.yaml \
    --ckpt runs/stage3_joint/best.pth --ood-root data/ood --out runs/protocol.json
python scripts/run_explainability.py --config configs/stage3_joint.yaml \
    --ckpt runs/stage3_joint/best.pth
python scripts/measure_efficiency.py --config configs/stage3_joint.yaml

# 4. Baselines — same splits, same degradations (FIRE first, it is the expensive one)
python scripts/train_baseline.py --config configs/baselines/npr.yaml
python scripts/train_baseline.py --config configs/baselines/cnndetection.yaml
python scripts/train_baseline.py --config configs/baselines/resnet50.yaml
```

Ablations are configs, not code paths:

```bash
python scripts/train.py --config configs/ablations/fusion_concat.yaml      # the mandatory one
python scripts/train.py --config configs/ablations/no_supcon.yaml
python scripts/train.py --config configs/ablations/no_band_entropy.yaml
python scripts/train.py --config configs/ablations/no_modality_dropout.yaml
python scripts/train.py --config configs/stage2_frequency.yaml --set model.frequency.n_bands=16
python scripts/train.py --config configs/stage3_joint.yaml \
    --set model.frequency.representation=magnitude_phase train.ckpt_dir=runs/abl_phase
```

## Three rules that are methodology, not housekeeping

1. **Crop, never resize, before the FFT.** Resampling imprints a kernel signature
   correlated with generator identity. `prepare.py` center-crops and reports how
   many images were too small to crop.
2. **Run the leak check before training.** If a classifier can tell generators
   apart from the spectra of *real* images, every cross-generator number is
   inflated.
3. **Never select checkpoints on in-distribution validation or on the test
   generators.** The validation generator exists for exactly this. `SplitSpec`
   raises if a generator appears in two roles.

## Layout

```
configs/            stage + ablation + baseline configs (YAML, _base_ inheritance)
ledd/data/          archive prep, splits, degradations, dataset, leak check
ledd/models/        fft rings, frequency stream, spatial stream, fusion, detector
ledd/losses/        SupCon (L_out + MoCo queue), band entropy, combined
ledd/engine/        3-stage trainer, evaluation protocol, metrics
ledd/explain/       Chefer relevance, band attribution, deletion/insertion, balance validation
ledd/baselines/     ResNet50 / CNNDetection / NPR + FIRE / DIRE adapters
scripts/            CLI entry points
tests/              pytest suite
```
