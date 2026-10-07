# LEDD — results ledger

*All aggregates recomputed from the raw per-epoch curves on 2026-10-07; seeded means,
standard deviations, last-5 means and robustness averages verified.*

Ground truth for the paper. Every number, how it was produced, and what qualifies it.
**Update this file as runs complete.** Prose can be rewritten; provenance cannot be
reconstructed.

Protocol throughout: GenImage, 160,000 images, 10k real + 10k fake per generator.
Train on sd_v14, sd_v15, wukong, vqdm, biggan, glide. **ADM = validation generator**
(model selection and all tuning). **Midjourney = test generator**, touched once.
AUC is the primary metric; accuracy and F1 only after threshold calibration.

---

## 1. Dataset controls

| Check | Result | Reading |
|---|---|---|
| Archive | 160,000 images, 8 generators, balanced, no cross-class duplicates | clean |
| Spectral leak check | **0.130** vs 0.125 chance (8-way, real images only) | pipeline does not encode generator identity |
| JPEG-history audit (pooled) | AUC 0.468, dev 0.032 | clean pooled |
| JPEG-history per generator | sd_v14 0.377 · sd_v15 0.328 · wukong 0.351 · vqdm 0.505 · biggan 0.514 · glide 0.544 · **adm 0.469** · **midjourney 0.668** | pooled value is misleading — opposite-direction biases cancel |

**Caveat to carry everywhere:** Midjourney, the test generator, has the strongest
blockiness bias (0.668). Every Midjourney number must be reported beside it. ADM, used
for all model selection, is clean (0.469).

`equalize_jpeg` was **off** when the archive was built. The audit says this is not
exploitable at pooled level, so the archive was not regenerated.

---

## 2. Main results

Cross-generator AUC on ADM unless stated. Seeds where available: mean ± std over 3.

| Model | Params | ADM (val) | Midjourney (test) |
|---|---|---|---|
| Frequency stream alone | <1M | 0.829 (in-dist 0.876, gap 0.047) | — |
| Spatial stream alone | 5.6M | 0.917 (in-dist 0.9998, gap 0.083) | — |
| Joint, cross-attention | 6.3M | best 0.933 ± 0.012 · mean 0.906 ± 0.001 | 0.963 |
| **Joint, concatenation** | **6.3M** | best **0.935 ± 0.003** · mean **0.923 ± 0.005** | **0.975** |

**Reported model: concatenation.** Cross-attention is tied on best-epoch, worse on the
honest (mean) measure, and markedly less stable: std of best-epoch across seeds is
**0.012 vs 0.003** (4x), and mean within-run swing is **0.081 vs 0.042** (2x). Note
concat seed 2 did have one 0.855 dip, so concat is not uniformly smooth — the stability
claim is about the distribution, not every run.

The single run that produced 0.9464 (joint, seed 42) is an outlier — seeds 1 and 2 gave
0.930 and 0.923. All Phase 3 evaluation originally computed from that checkpoint was
redone on concat.

### Calibrated metrics (threshold fitted on ADM only, T = −7.52)

| Split | AUC | Acc | F1 |
|---|---|---|---|
| In-distribution | 0.9996 | 0.905 | 0.913 |
| ADM (validation) | 0.9365 | 0.854 | 0.860 |
| Midjourney (test) | 0.9750 | 0.894 | 0.902 |

Uncalibrated F1 on ADM was 0.550. Calibration changes no model, only the threshold.

---

## 3. Baselines

Same splits, preprocessing and degradations. Single seed each.

| Model | Params | ADM | Midjourney | F1 (test, calibrated) |
|---|---|---|---|---|
| NPR | 1.44M | 0.931 | 0.905 | 0.831 |
| CNNDetection | 25M | 0.924 | 0.964 | 0.886 |
| ResNet50 | 25M | 0.936 | 0.953 | 0.879 |
| **LEDD (concat)** | **6.3M** | 0.935 | **0.975** | **0.902** |

**Outstanding:** CNNDetection and ResNet50 were trained before the optimiser fix, with
every parameter at `lr_new` = 1e-3 (too high for a pretrained backbone). Re-runs pending
as `base_cnndetection2`, `base_resnet502`. NPR trains from scratch, so unaffected.

---

## 4. Robustness (Midjourney, degraded)

| Model | Clean | JPEG-75 | JPEG-50 | Blur σ2 | Resize ½ | Mean degraded | Worst |
|---|---|---|---|---|---|---|---|
| NPR | 0.905 | 0.819 | 0.803 | 0.816 | **0.605** | 0.761 | 0.605 |
| CNNDetection | 0.964 | 0.937 | 0.912 | **0.886** | 0.799 | 0.884 | 0.799 |
| ResNet50 | 0.953 | 0.924 | 0.899 | 0.815 | 0.881 | 0.880 | 0.815 |
| **LEDD (concat)** | **0.975** | **0.945** | **0.916** | 0.868 | **0.912** | **0.910** | **0.868** |

Wins 4 of 5 conditions, the degraded mean by 0.026, and the worst case by 0.053. The only
loss is blur, to CNNDetection — whose method *is* blur augmentation.

**Novel finding:** NPR collapses to **0.605 under resizing**, near chance. Its mechanism
reads neighbouring-pixel relationships left by up-sampling, which resampling destroys.
Tan et al. (CVPR 2024) report no robustness experiments; this supplies the missing
evaluation and explains the failure mechanistically.

---

## 5. Ablations

All at 10 epochs, **single seed (42)** — unlike the seeded main table in S2. Mean =
average over all epochs, which is the honest measure: best-epoch selection rewards
whichever run was noisiest.

| Setting (seed 42) | Best | Mean | Swing | Verdict |
|---|---|---|---|---|
| Cross-attention fusion | 0.946 | 0.905 | 0.097 | — |
| **Concatenation** | 0.937 | **0.928** | 0.017 | **simpler wins** |
| − modality dropout | 0.929 | 0.903 | 0.053 | no help; only adds variance |
| − contrastive loss | 0.934 | 0.886 | 0.298 | no help |
| − band-entropy | 0.941 | **0.913** | 0.091 | no help (slightly better without) |

Cross-check: with three seeds (S2) the fusion comparison is concat 0.923 ± 0.005 vs
cross-attention 0.906 ± 0.001, non-overlapping. The single-seed ablations above are
directional only.

**Frequency representation** (frequency stream alone, last-5-epoch mean):
magnitude 0.820 · **DCT 0.822** · magnitude+phase 0.810.
Magnitude and DCT equivalent; **adding phase hurts**.

**Band count** (same measure): 8 → 0.814 · **12 → 0.820** · 16 → 0.804.
Twelve confirmed optimal — too few loses resolution, too many dilutes each band.

Four of four added components fail to help. Two design choices are validated by data.

---

## 6. Explanation faithfulness

Deletion: lower is better. Insertion: higher is better. Random-saliency control included.

| Explanation | Result | Verdict |
|---|---|---|
| Spatial maps — insertion | 0.765 vs random 0.566 | **faithful** |
| Spatial maps — deletion | 0.625 vs random 0.577 | fails; a masked patch is itself smooth, which a synthetic-image detector reacts to |
| Stream balance metric | claimed 97% frequency vs **causal 4%** | **unfaithful — report causal only** |

### Attention is not explanation (three demonstrations)

1. **Band level, frequency stream alone.** Attention peaks mid-band (56% of mass on real
   images, rings 4–5, radius 0.58–0.71), matching FIRE's published claim. Causal band
   deletion disagrees: **band 11 (radius 0.96–1.00) costs 0.270 AUC** (0.830 → 0.560) and
   band 1 costs 0.156, while the mid-band rings cost ~0.00.
2. **Stream level, joint model.** Attention claims 0.974 frequency reliance; stream
   deletion shows 0.041.
3. **Band level, concat model.** No band costs more than 0.003.

### The frequency stream is informative alone but redundant in combination

Stream deletion on the reported concat model (ADM, baseline 0.9332):

| Removed | AUC | Drop |
|---|---|---|
| nothing | 0.9332 | — |
| **frequency stream** | 0.9300 | **0.0031** |
| spatial stream | 0.8084 | 0.1247 |

The frequency stream alone still scores 0.808, so it carries real signal — but the
MobileViT backbone already encodes it. **Explicit frequency modelling is redundant with a
modern hybrid CNN-transformer backbone**, contradicting the assumption underlying F³-Net,
FreqCross and FIRE. No prior work measures this, because no prior work runs stream
deletion.

Consequence for the paper: report band attribution for the **standalone frequency stream**
(a <1M-parameter, 0.829-AUC detector where band 11 causally matters), not for the joint
model.

---

## 7. Efficiency

6,306,633 parameters. No diffusion model and no large frozen encoder at inference.
FLOPs via fvcore **undercount** — `scaled_dot_product_attention` and `fft_fft2` are
unsupported operators, so report as "conv+linear FLOPs" or cross-check with thop.

---

## 8. Training behaviour worth reporting

**Cross-generator accuracy peaks early then declines.** The spatial stream reaches its
best ADM score at **epoch 1** and falls afterwards while in-distribution AUC climbs to
0.9998. Selecting on in-distribution validation would have cost 0.025 AUC. Direct evidence
that the dedicated validation generator was necessary.

**SupCon dominates the objective.** At λ=0.5 the contributions are BCE 0.17 vs SupCon
4.33 — roughly 1:25, not the configured 1:0.5. Report the effective ratio.

---

## 9. Open items

| Item | Status |
|---|---|
| `ctrl_spatial_from_s1` — spatial-only, matched init and schedule | **pending; closes the dual-stream claim** |
| CNNDetection / ResNet50 re-run with fixed optimiser | pending (fairness) |
| OOD set: SDXL, SD3, Flux + COCO reals | **pending; the last major gap** |
| FIRE baseline | pending (6–10 h) |
| Baseline seeds | single seed only; ours has three |

---

## 10. Claims the paper can make

1. A 6.3M-parameter detector matches 25M baselines on clean images and **beats all of them
   on degraded images** (mean 0.910 vs 0.884; worst case 0.868 vs 0.815).
2. **Attention-based explanations are unfaithful**, shown three independent ways.
3. **Frequency modelling is redundant** with a modern spatial backbone despite being
   informative in isolation.
4. **NPR fails under resizing** (0.605), an unreported weakness of a CVPR 2024 method.
5. Evaluation controls that are not standard in this literature: a spectral leak check, a
   per-generator compression-bias audit, a three-role generator split, seeded ablations
   reported by mean rather than best epoch, and threshold calibration fitted off the test
   split.

## Claims the paper cannot make

- That cross-attention fusion beats concatenation. It does not.
- That the contrastive loss, band-entropy regulariser or modality dropout help. None do.
- That dual-stream beats single-stream — pending `ctrl_spatial_from_s1`, and the 0.003
  stream-deletion result suggests it will not.
