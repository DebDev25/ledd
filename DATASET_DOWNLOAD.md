# Downloading GenImage — step by step

Do this **after** Phase 0 and Phase 1 of `COLAB_SETUP.md` pass. All of it runs
inside Colab or Kaggle; nothing needs to touch your laptop.

---

## Where the data actually lives

The official release (<https://github.com/GenImage-Dataset/GenImage>) is distributed
via **Baidu Yunpan**, which is painful outside China. Use a mirror instead:

| Source | Coverage | Notes |
|---|---|---|
| **Kaggle mirror** — `vtphatt2/genimage-stable-diffusion-v1-4` | SD v1.4 subset | Easiest start; attaches straight into a Kaggle notebook |
| **GenImage-mirror index** — <https://github.com/vtphatt2/GenImage-mirror> | links to the other subsets | Check here for the remaining seven generators |
| Hugging Face community mirrors (`jzousz/GenImage`, `blorg469/genimage`) | partial / varying | Verify contents before trusting; community re-uploads are not always complete |
| Official Baidu | all 8 | Last resort |

Subset names you need, and their native resolutions:

`sd_v14` (512) · `sd_v15` (512) · `wukong` (512) · `midjourney` (1024) ·
`adm` (256) · `glide` (256) · `vqdm` (256) · `biggan` (128)

Each subset unpacks as `<generator>/imagenet_ai_*/train/{nature,ai}/` —
`nature` = real, `ai` = fake.

---

## Step 1 — Get ONE generator first (do not download all eight)

### Option A: Kaggle notebook (recommended)

1. kaggle.com → **Create → New Notebook**
2. Right panel → **Add Input** → search `genimage stable diffusion v1-4` → Add
3. The data appears read-only at `/kaggle/input/genimage-stable-diffusion-v1-4/`
4. Settings → Accelerator → **GPU T4 x2**

No download, no disk cost, available every session.

### Option B: Colab via the Kaggle API

```python
# upload kaggle.json (kaggle.com -> Settings -> API -> Create New Token) via the file pane
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!pip -q install kaggle
!kaggle datasets download -d vtphatt2/genimage-stable-diffusion-v1-4 -p /content/raw --unzip
```

**Never commit `kaggle.json`** — the repo is public. `.gitignore` covers it, but
upload it through the file pane rather than putting it in the project folder.

---

## Step 2 — Audit the raw data before preprocessing

```python
!python scripts/audit_formats.py \
    --real /content/raw/imagenet_ai_0419_sdv4/train/nature \
    --fake /content/raw/imagenet_ai_0419_sdv4/train/ai
```

This checks a known trap: GenImage's real images are ImageNet **JPEGs** while
several fake subsets are **PNG**. Ricker et al. (*"Fake or JPEG? Revealing Common
Biases in Generated Image Detection Datasets"*, arXiv:2403.17608) showed detectors
exploit that difference rather than learning generator artifacts. Saving everything
as PNG does **not** fix it — the reals already carry baked-in JPEG artifacts.

`prepare_data.py --equalize-jpeg 95` (the default) re-encodes *both* classes at the
same quality so they share an identical compression history. Report the setting in
the paper.

---

## Step 3 — Build the crop archive

```python
!python scripts/prepare_data.py \
    --src /content/raw --dst data/genimage_224 \
    --generators sd_v14 --n-per-class 2000 \
    --layout "imagenet_ai_0419_sdv4/{split}/{label}" --split train
```

Adjust `--layout` to match your mirror's folder names (it prints what it cannot
find). Watch the "upscaled" percentage in the summary — images smaller than 224 had
to be resized, and if that number is non-trivial you must report it.

## Step 4 — Leak check, before any GPU time

```python
!python scripts/run_leak_check.py --archive data/genimage_224 --generators sd_v14
```

With one generator this is trivially clean; it becomes meaningful at step 6.

## Step 5 — Prove the pipeline learns on real data

Train the frequency stream alone — minutes, not hours:

```python
!python scripts/train.py --config configs/stage2_frequency.yaml \
    --set data.root=data/genimage_224 train.epochs=3 \
         data.train_generators="[sd_v14]" data.val_generator=sd_v14 \
         data.test_generators="[sd_v14]" train.ckpt_dir=runs/probe_sd14
```

AUC should climb clearly above 0.5. If it sits at chance while the synthetic run
worked, the problem is data or preprocessing — and you have found out in twenty
minutes instead of two days.

## Step 6 — Scale to all eight

Repeat steps 1–3 per generator, **deleting each raw download immediately after
preprocessing** so peak disk stays near 20 GB rather than 52 GB:

```python
for gen in ["sd_v15", "wukong", "vqdm", "biggan", "midjourney", "adm", "glide"]:
    # download -> prepare_data.py --generators {gen} --n-per-class 10000 -> rm -rf raw/{gen}
    ...
```

Then run the leak check across all eight — this is where it earns its keep:

```python
!python scripts/run_leak_check.py --archive data/genimage_224 \
    --generators sd_v14 sd_v15 wukong vqdm biggan midjourney adm glide
```

Non-zero exit = the preprocessing pipeline leaks generator identity. Fix it before
training; every cross-generator number depends on this.

## Step 7 — Host the finished archive

Upload the ~15–18 GB archive once so no session ever rebuilds it:

```python
!kaggle datasets init -p data/genimage_224
# edit dataset-metadata.json (title, id: <your-kaggle-user>/genimage-224-ledd)
!kaggle datasets create -p data/genimage_224 --dir-mode zip
```

Mirror it to Hugging Face as a backup. From then on, attach it as a Kaggle input and
skip steps 1–3 entirely.

---

## The OOD set (week 4, ~4–6 GB)

Not a download — you generate it.

- **Reals:** COCO val2017, `!wget http://images.cocodataset.org/zips/val2017.zip` (~1 GB)
- **Fakes:** 300–500 images each from SDXL, SD3 and Flux via free tiers, Colab, or API credits. Prompt them from COCO captions so content distribution roughly matches the reals.
- Layout: `data/ood/<sdxl|sd3|flux>/{real,fake}/*.png`
- Run them through the **same** `prepare_data.py` settings, including `--equalize-jpeg 95`. A different preprocessing path for the OOD set invalidates the comparison.
