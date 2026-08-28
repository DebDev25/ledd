# Baselines

All baselines must be trained and evaluated on **the same splits, the same
preprocessing and the same degradations** as LEDD, or the comparison is
meaningless. Use `scripts/train_baseline.py`, which reuses the same split file.

| Baseline | Role | Status | How |
|---|---|---|---|
| ResNet50 | standard anchor | reimplemented | `python scripts/train_baseline.py --baseline resnet50 --config configs/baselines/resnet50.yaml` |
| CNNDetection (Wang 2020) | foundational | reimplemented (their augmentation recipe) | `--baseline cnndetection --config configs/baselines/cnndetection.yaml` |
| NPR (Tan CVPR 2024) | lightweight comparison, 1.44M params | reimplemented | `--baseline npr --config configs/baselines/npr.yaml` |
| **FIRE (Chu CVPR 2025)** | SOTA + frequency competitor | external repo | see below |
| DIRE (ICCV 2023) | historic diffusion baseline | inference-only | see below |
| DistilDIRE / UnivFD | context | cite reported numbers | — |

## FIRE — schedule this first

FIRE trains end-to-end through a frozen LDM autoencoder (two encoder/decoder passes
per image), so it is the **most expensive item in the whole project** — budget 3–5×
LEDD's training cost. Start it before anything else in the baseline phase so a
surprise here does not eat the model phase.

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/Chuchad/FIRE.git
# follow their README for the LDM autoencoder weights
```

Then train with their default config on our subset (reduce to ~40–60k images at
their 256 resolution if compute is tight — **record the subset size in the paper**),
and evaluate through our harness:

```bash
python scripts/evaluate.py --external fire --repo third_party/FIRE \
       --ckpt third_party/FIRE/weights/fire.pth --splits runs/stage3_joint/splits.json
```

Citation note: FIRE is **CVPR 2025**, not an arXiv preprint — fix this in the
literature summaries and Related Work.

## DIRE — inference only

Per-image DDIM inversion (S=20 steps by default) makes full training infeasible on
free-tier GPUs. Run their released checkpoint over the test sets only
(~20–30k images, feasible overnight), or cite the published numbers. Note in the
paper that DIRE collapses to near-chance on SDXL-era generators (FIRE's own
results), which is precisely why FIRE replaced it as the primary comparison.
