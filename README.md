# LEDD — Lightweight Explainable Diffusion Detection

Dual-stream detector for diffusion-generated images: MobileViT-S on RGB + a
radial band-token frequency stream, fused by bidirectional cross-attention with a
`[FUSE]` readout. ~7M parameters, intrinsic frequency-band attribution, causal
faithfulness evaluation.

See `../Architecture_Methodology_v2.docx` for the design rationale behind every
choice here.

## Status

| Component | State |
|---|---|
| Data pipeline, leak check, degradations | code complete, tested |
| Streams, fusion, detector | code complete, tests written |
| Losses (BCE + SupCon + band entropy) | code complete, tests written |
| Training engine (3 stages, resume, AMP) | code complete |
| Explainability + faithfulness | code complete, tests written |
| Baselines (ResNet50 / CNNDetection / NPR) | code complete |
| Baselines (FIRE / DIRE) | adapters + instructions, external repos |

**Nothing has been run on a GPU yet.** Config/data tests (13) pass; the
torch-dependent tests are written but need a machine with torch installed.

**Start here:** `COLAB_SETUP.md` — phased setup that verifies the code on synthetic
data before you download a single GB. Run `pytest tests/ -q` then
`python scripts/smoke_test.py --cuda` first thing.

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
