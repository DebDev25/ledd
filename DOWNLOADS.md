# What to download

Nothing here has been downloaded — the code expects these paths. Total disk:
**~60 GB** during preprocessing, **~25 GB** steady state.

## 1. GenImage (required, the big one)

Official: <https://github.com/GenImage-Dataset/GenImage> (Baidu; mirrors on
Hugging Face and Kaggle). Do **not** pull all ~500 GB–1 TB.

Download only these 8 generator subsets, and only their `train` split:

| Generator | Config name | Native res | Approx raw size for 10k+10k |
|---|---|---|---|
| Stable Diffusion v1.4 | `sd_v14` | 512 | ~5 GB |
| Stable Diffusion v1.5 | `sd_v15` | 512 | ~5 GB |
| Wukong | `wukong` | 512 | ~5 GB |
| VQDM | `vqdm` | 256 | ~2 GB |
| BigGAN | `biggan` | 128 | ~1 GB |
| Midjourney | `midjourney` | 1024 | ~12 GB |
| ADM | `adm` | 256 | ~2 GB |
| GLIDE | `glide` | 256 | ~2 GB |

Sampling is random within each subset, so you can grab a partial download and
still get a valid 10k sample — but grab reals and fakes from the *same* subset.

After `prepare_data.py` the archive is **~15–18 GB** and the raw downloads can be
deleted. Upload that archive as a **Kaggle Dataset** so notebooks attach it
instantly instead of re-downloading each session. Mirror it on Hugging Face Hub.

Expected layout before preprocessing (GenImage's own):
```
raw/GenImage/<generator>/imagenet_ai_.../train/{nature,ai}/*.{jpg,png}
```
`prepare_data.py --layout` takes a pattern if your mirror differs.

## 2. OOD test set (required, small — build it yourself)

A few hundred images each from **SDXL, SD3, Flux**, plus real counterparts from
**COCO val2017** (<https://cocodataset.org/#download>, ~1 GB) or a LAION subset.
Generate via free tiers / Colab / API credits. Target layout:

```
data/ood/<sdxl|sd3|flux>/{real,fake}/*.png
```

Total ~4–6 GB. Build this **early** — reviewers will ask, and GenImage predates
all three generators.

## 3. Pretrained weights (automatic)

- **MobileViT-S** — `timm` downloads `mobilevit_s` ImageNet weights on first use (~22 MB).
- **ResNet50** — torchvision downloads on first use (~100 MB).

Nothing to fetch manually.

## 4. Baseline repos (clone when you reach the baseline phase)

| What | Where | Notes |
|---|---|---|
| **FIRE** | `git clone https://github.com/Chuchad/FIRE third_party/FIRE` | CVPR 2025, MIT. Also needs their LDM autoencoder weights (~320 MB). **Most expensive item in the project — start it first.** |
| DIRE | <https://github.com/ZhendongWang6/DIRE> | Released checkpoint + ADM weights (~2 GB). Inference-only on test sets. |
| CNNDetection | <https://github.com/PeterWang512/CNNDetection> | Optional — we reimplement their recipe; clone only to run their released weights. |
| NPR | <https://github.com/chuangchuangtan/NPR-DeepfakeDetection> | Optional — reimplemented in `ledd/baselines/simple.py`. |

## 5. Python packages

`pip install -r requirements.txt` — torch, torchvision, timm, numpy, Pillow,
PyYAML, scikit-learn, tqdm, fvcore, thop, matplotlib, pytest. On Kaggle/Colab most
are preinstalled; `timm`, `fvcore` and `thop` usually are not.

## Disk budget

| Stage | Peak |
|---|---|
| Raw GenImage subsets (transient) | ~34 GB |
| Crop archive (keep) | ~15–18 GB |
| OOD set (keep) | ~4–6 GB |
| Checkpoints, logs, baseline weights | ~5–10 GB |
| **Steady state after deleting raw** | **~25–35 GB** |
