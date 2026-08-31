# Running on Kaggle — from uploaded archive to first real results

Dataset is uploaded. Work through these in order; each step gates the next.

---

## 1. Notebook setup (5 min)

kaggle.com → **Create → New Notebook**, then in the right panel:

- **Add Input** → your dataset → note the mount path, e.g. `/kaggle/input/genimage-224-ledd`
- **Accelerator** → `GPU T4 x2` (or P100 — one GPU is used either way)
- **Persistence** → *Files only* (keeps `/kaggle/working` between sessions)
- **Internet** → On (needed to clone the repo and fetch timm weights)

```python
!git clone https://github.com/<your-user>/ledd.git /kaggle/working/ledd
%cd /kaggle/working/ledd
!pip -q install timm fvcore thop
!python -m pytest tests/ -q && python scripts/smoke_test.py --cuda
```

Point the config at your dataset — edit `configs/kaggle.yaml` or override inline:

```python
DATA = "/kaggle/input/genimage-224-ledd"     # your actual mount path
```

---

## 2. Verify the archive (2 min) — do this before anything else

```python
!python scripts/verify_archive.py --archive {DATA}
```

Checks counts, class balance, image size, formats, duplicates across
classes/generators, and whether `equalize_jpeg` was recorded. Non-zero exit means
fix the archive first — every number you produce afterwards depends on it.

## 3. Leak check (5 min) — the real gate

```python
!python scripts/run_leak_check.py --archive {DATA} \
    --generators sd_v14 sd_v15 wukong vqdm biggan glide adm midjourney
```

Trains a classifier to predict *generator identity from the spectra of REAL images
only*. Real images are content-comparable across subsets, so above-chance accuracy
means your preprocessing pipeline — not the generators — is discriminable, and
every cross-generator result would be inflated.

- accuracy ≈ 1/8 = 0.125 → clean, proceed
- accuracy ≫ 0.125 → stop and fix preprocessing

**Save this number.** It belongs in the paper as evidence of a controlled pipeline;
almost no detection paper reports it.

---

## 4. Note the generator split (it changed)

I corrected `configs/default.yaml`:

| Role | Generators | Why |
|---|---|---|
| Train | sd_v14, sd_v15, wukong, vqdm, biggan, **glide** | SD-family + diverse architectures |
| Validation | **adm** | Moderately dissimilar — checkpoint selection and tuning only |
| Test | **midjourney** | GenImage's correlation analysis makes it the hardest transfer target |

Midjourney was previously in training, which wasted your hardest test case. Do not
look at Midjourney numbers until the very end — that is the whole point of the
three-role split.

---

## 5. Stage 2 first — the frequency stream (~30 min)

Train the cheap stream before the expensive one. Under 1M parameters, minutes per
epoch, and it tells you immediately whether the frequency signal is real.

```python
!python scripts/train.py --config configs/kaggle.yaml \
    --set stage=frequency data.root={DATA} train.epochs=15 train.batch_size=256 \
         loss.supcon_weight=0.0 loss.band_entropy_weight=0.0 \
         train.ckpt_dir=/kaggle/working/runs/stage2_frequency
```

**What to expect:** validation-generator AUC meaningfully above 0.5 — likely
0.65–0.85. Diffusion images lack the obvious GAN grid artifacts, so do not expect
0.99; the frequency stream is one half of the argument, not the whole detector.

If it sits at 0.5, stop and diagnose — dataloader labels, or a preprocessing fault.

Then look at what it learned:

```python
!python scripts/run_explainability.py --config configs/kaggle.yaml \
    --ckpt /kaggle/working/runs/stage2_frequency/best.pth --n-batches 4
```

Check which bands the attention concentrates on. **If it favours mid-band rings,
you have independently reproduced FIRE's central finding with a completely
different mechanism** — that is a result worth a paragraph in the paper either way.

## 6. Stage 1 — the spatial stream (~2–3 h)

```python
!python scripts/train.py --config configs/kaggle.yaml \
    --set stage=spatial data.root={DATA} train.epochs=12 train.batch_size=96 \
         train.lr_backbone=1e-4 loss.supcon_weight=0.0 loss.band_entropy_weight=0.0 \
         train.ckpt_dir=/kaggle/working/runs/stage1_spatial
```

Expect higher in-distribution AUC than Stage 2 and a *larger* gap between
in-distribution and validation-generator AUC — the spatial stream overfits to
generator fingerprints, which is exactly the weakness the frequency stream and the
contrastive loss exist to offset. Record both numbers; that gap is a figure.

## 7. Stage 3 — joint (~3–4 h)

```python
!python scripts/train.py --config configs/kaggle.yaml \
    --set data.root={DATA} \
         init.spatial_ckpt=/kaggle/working/runs/stage1_spatial/best.pth \
         init.frequency_ckpt=/kaggle/working/runs/stage2_frequency/best.pth \
         train.ckpt_dir=/kaggle/working/runs/stage3_joint
```

Full loss, modality dropout, cross-attention fusion. The number that matters is
**validation-generator AUC vs the two single-stream runs**. If joint does not beat
both, the fusion is not earning its parameters — tell me and we diagnose before you
run ablations.

## 8. Evaluate + explain + measure

```python
!python scripts/evaluate.py --config configs/kaggle.yaml \
    --ckpt /kaggle/working/runs/stage3_joint/best.pth \
    --out /kaggle/working/runs/protocol.json
!python scripts/run_explainability.py --config configs/kaggle.yaml \
    --ckpt /kaggle/working/runs/stage3_joint/best.pth
!python scripts/measure_efficiency.py --config configs/kaggle.yaml \
    --ckpt /kaggle/working/runs/stage3_joint/best.pth
```

## 9. The one ablation to run immediately

Everything rests on cross-attention beating concatenation. Run it before the other
ablations, because a null result changes the paper:

```python
!python scripts/train.py --config configs/ablations/fusion_concat.yaml \
    --set data.root={DATA} \
         init.spatial_ckpt=/kaggle/working/runs/stage1_spatial/best.pth \
         init.frequency_ckpt=/kaggle/working/runs/stage2_frequency/best.pth \
         train.ckpt_dir=/kaggle/working/runs/abl_concat
```

---

## Surviving Kaggle sessions

- GPU sessions cap at ~12 h and the weekly quota is 30 h; every stage above fits in one session.
- With **Persistence: Files only**, `/kaggle/working` survives between sessions, so `train.py` resumes automatically from `last.pth`.
- For safety across quota resets, save checkpoints as a Kaggle Dataset output at the end of a big run and attach it as an input next time.
- Three accounts ≈ 90 GPU-h/week. Split the work: one person Stage 1, one Stage 2, one the baselines — the streams are independent until Stage 3.

## Order, one line

```
verify_archive → leak_check → Stage 2 → Stage 1 → Stage 3 → evaluate/explain/efficiency → concat ablation → baselines
```
