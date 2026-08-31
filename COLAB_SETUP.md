# Colab / Kaggle setup and first test run

Work through the phases in order. **Phase 0 needs no dataset at all** — do not
download GenImage until Phase 0 and Phase 1 both pass.

---

## Phase 0 — Smoke test (no data, ~15 min)

### 0.1 Push the code to GitHub (once, from your laptop)

```bash
cd "D:\Important work\VIT Academics\Project 1\Papers\Summaries\ledd"
git init
git add .
git commit -m "LEDD: initial architecture, losses, explainability, tests"
git branch -M main
git remote add origin https://github.com/<your-user>/ledd.git
git push -u origin main
```

The repo is **public**, so no tokens are needed anywhere — every teammate clones the
same way, and Colab/Kaggle need no credentials at all. Add your two teammates as
collaborators (Settings → Collaborators) so they can push. GitHub is how the code
reaches Colab and Kaggle — do not upload zips by hand, you will lose track of which
version ran.

Because it is public, never commit anything secret: no `kaggle.json`, no Hugging Face
token, no API keys. `.gitignore` already excludes `data/`, `runs/` and `*.pth`.

### 0.2 New Colab notebook → Runtime → Change runtime type → **T4 GPU**

### 0.3 Cell 1 — environment

```python
!nvidia-smi
import torch; print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
!pip -q install timm fvcore thop
```

Colab ships torch and torchvision; `timm` (MobileViT weights), `fvcore` and `thop`
(FLOPs) are the ones you must add.

### 0.4 Cell 2 — get the code

```python
!git clone https://github.com/<your-user>/ledd.git
%cd /content/ledd
```

Public repo, so no token and no login. To pull your latest changes in a later
session, `%cd /content/ledd` then `!git pull`.

Pushing *from* Colab does still need credentials, so treat Colab as read-only:
edit locally in PyCharm, push from there, `git pull` in Colab.

### 0.5 Cell 3 — unit tests

```python
!pip -q install pytest
!python -m pytest tests/ -q
```

Expect ~45 passed. **If anything fails, stop and fix before continuing** — these
have never run on a machine with torch.

### 0.6 Cell 4 — smoke test

```python
!python scripts/smoke_test.py --cuda
```

This is the important one. It exercises the paths flagged as unverified:
MobileViT token-stage indexing, AMP + attention recording, Chefer relevance
shapes, band masking, deletion/insertion. Expect a PASS table and
`11/11 checks passed`.

Most likely first failure is `model builds` or `forward pass` — a MobileViT
feature-list mismatch. Re-run with tracebacks:

```python
!SMOKE_VERBOSE=1 python scripts/smoke_test.py --cuda
```

### 0.7 Cell 5 — full-size sanity + efficiency

```python
!python scripts/smoke_test.py --cuda --full-size
!python scripts/measure_efficiency.py --config configs/stage3_joint.yaml
```

`--full-size` builds at 224 with real ImageNet weights. The efficiency report is
your first real paper number: parameters should land near **7M**.

---

## Phase 1 — Pipeline run on synthetic data (~20 min, still no download)

Proves the training loop, splits, checkpointing, evaluation and explainability all
work together — before a single GB is downloaded.

```python
# Cell 6 — 960 synthetic images (~30 s)
!python scripts/make_dummy_data.py --dst data/dummy_224 --n 60

# Cell 7 — two epochs through the full joint pipeline
!python scripts/train.py --config configs/smoke.yaml

# Cell 8 — evaluation protocol + explainability
!python scripts/evaluate.py --config configs/smoke.yaml --ckpt runs/smoke/best.pth
!python scripts/run_explainability.py --config configs/smoke.yaml \
        --ckpt runs/smoke/best.pth --n-batches 2 --batch-size 8
```

The dummy "fakes" are half-resolution images upsampled with nearest neighbour —
the same up-sampling artifact real decoders leave — so AUC should climb above 0.5.
**These numbers are meaningless scientifically.** You are checking that the loop
runs, checkpoints resume, and the JSON reports are produced.

Then verify resume works, because you will depend on it constantly:

```python
# Cell 9 — kill and resume
!python scripts/train.py --config configs/smoke.yaml --set train.epochs=4
```

It should log `resumed from runs/smoke/last.pth @ epoch 2` and run only 2 more.

---

## Phase 2 — Real data

Only now download anything. See `DOWNLOADS.md` for the full table; the short list:

| # | What | Size | When |
|---|---|---|---|
| 1 | GenImage — 8 generator subsets, train split only | ~34 GB raw → ~15–18 GB after prep | Now |
| 2 | COCO val2017 (OOD reals) | ~1 GB | Week 4 |
| 3 | SDXL / SD3 / Flux images you generate | ~3–5 GB | Week 4 |
| 4 | FIRE repo + LDM autoencoder weights | ~350 MB | Baseline phase |
| 5 | MobileViT-S, ResNet50 weights | ~120 MB | automatic |

### 2.1 Start with ONE generator

Do not download all eight. Take `sd_v14` only, build the archive, and run the leak
check:

```python
!python scripts/prepare_data.py --src /content/raw/GenImage --dst data/genimage_224 \
        --generators sd_v14 --n-per-class 2000
!python scripts/run_leak_check.py --archive data/genimage_224 --generators sd_v14
```

### 2.2 Then a 2-generator training run

Train Stage 2 (the frequency stream — minutes, not hours) on two generators before
committing to the full download. If AUC is at chance on real data while the dummy
run worked, the problem is data or preprocessing, and you have spent 20 minutes
finding out instead of two days.

### 2.3 Only then scale up

Download the remaining generators, build the full archive, **upload it as a Kaggle
Dataset**, and move training to Kaggle (30 published GPU-h/week per account vs
Colab's unpublished quota). Keep Colab for quick experiments.

---

## Persistence, because sessions die

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/ledd_runs
!ln -sfn /content/drive/MyDrive/ledd_runs /content/ledd/runs
```

Run this **before** training. Checkpoints then land on Drive, and a killed session
resumes with the same `train.py` command. Never put the dataset on Drive — it is
15 GB against a 15 GB quota, and Drive I/O is slow.

## Order of operations, one line each

```
pytest → smoke_test → dummy data → smoke config train → evaluate → explain
→ 1 generator → leak check → 2-generator Stage 2 → full archive → Kaggle
```
