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
| Split integrity (hashed across all 25 runs) | `val_generator` 1 hash (n=20,000) · `test_generator` 1 hash (n=20,000) · `train`/`val_indist` 3 hashes, one per seed | clean — see note |
| Spectral leak check | **0.130** vs 0.125 chance (8-way, real images only) | pipeline does not encode generator identity |
| JPEG-history audit (pooled) | AUC 0.468, dev 0.032 | clean pooled |
| JPEG-history per generator | sd_v14 0.377 · sd_v15 0.328 · wukong 0.351 · vqdm 0.505 · biggan 0.514 · glide 0.544 · **adm 0.469** · **midjourney 0.668** | pooled value is misleading — opposite-direction biases cancel |

**Caveat to carry everywhere:** Midjourney, the test generator, has the strongest
blockiness bias (0.668). Every Midjourney number must be reported beside it. ADM, used
for all model selection, is clean (0.469).

`equalize_jpeg` was **off** when the archive was built. The audit says this is not
exploitable at pooled level, so the archive was not regenerated.

**Why `train` has three hashes and that is not a problem.** `make_splits` seeds only
`rng.shuffle(train_items)`, which carves the 5% in-distribution holdout out of the training
generators. The generator *roles* come from the config, and `val_generator` / `test_generator`
are built by `build_index()` — a sorted directory listing with no RNG. So changing the seed
repartitions train/val_indist and nothing else: **all 25 runs are scored on the same 20,000
ADM and 20,000 Midjourney images**. Counts check: 114,000 + 6,000 = 120,000 = 6 x 20,000.

Consequence worth stating in the paper: the seeded standard deviations capture
data-partition variance as well as initialisation variance, which strengthens the 3.7-sigma
control rather than weakening it. The only seed-dependent quantities are in-distribution AUCs
(0.9998 spatial, 0.9889 ResNet-50 epoch 0), both single-seed and used qualitatively in S8.

---

## 2. Main results

Cross-generator AUC on ADM unless stated. Seeds where available: mean ± std over 3.
**Gap is defined as in-distribution minus cross-generator AUC at the SAME (selected)
epoch** — not against the in-distribution maximum, which would inflate it. Against the
maximum the gaps read 0.047 and 0.083; the like-for-like figures are 0.042 and 0.079.
The spatial stream's gap is **1.88x** the frequency stream's either way.

| Model | Params | ADM (val) | Midjourney (test) |
|---|---|---|---|
| Frequency stream alone | <1M | 0.829 (in-dist 0.871 at the same epoch, **gap 0.042**) | — |
| Spatial stream alone | 5.6M | 0.917 (in-dist 0.996 at the same epoch, **gap 0.079**) | — |
| Joint, cross-attention | 6.3M | best 0.933 ± 0.012 · mean 0.906 ± 0.001 | 0.963 |
| **Joint, concatenation** | **5.6M** | best **0.935 ± 0.003** · mean **0.923 ± 0.005** | **0.975** |

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

All baselines retrained with the corrected optimiser (backbone 1e-4, head 1e-3).

| Model | Params | ADM (val) | Midjourney (test) | F1 (test) | MJ − ADM |
|---|---|---|---|---|---|
| NPR | 1.44M | 0.931 | 0.905 | 0.831 | −0.026 |
| CNNDetection | 25.6M | 0.927 | 0.963 | 0.883 | +0.037 |
| ResNet50 | 25.6M | **0.945** | 0.890 | 0.795 | **−0.055** |
| **LEDD (concat)** | **5.6M** | 0.935 | **0.975** | **0.902** | +0.040 |

Pre-fix values for the record: CNNDetection ADM 0.924 / MJ 0.964; ResNet50 ADM 0.936 /
MJ 0.953. The fix raised both on ADM and lowered ResNet50 on MJ.

### ResNet50 is a selection-rule pathology worth reporting

ResNet50 wins the validation generator and loses the test generator by 0.085. Its best
ADM epoch was **epoch 0**, where in-distribution AUC was only 0.9889 — the model was
barely trained. Best-epoch selection on a single validation generator picked an
undertrained checkpoint that happened to suit ADM and transfers poorly.

**This is a limitation of our own protocol, not only of the baseline**, and should be
stated as such. LEDD's selected epochs were 8, 3 and 6 (converged), so it was unaffected.
A more robust rule restricts selection to epochs where in-distribution AUC has plateaued.
Baselines remain single-seed; LEDD has three.

---

## 4. Robustness (Midjourney, degraded)

| Model | Clean | JPEG-75 | JPEG-50 | Blur σ2 | Resize ½ | Mean degraded | Worst |
|---|---|---|---|---|---|---|---|
| NPR | 0.905 | 0.819 | 0.803 | 0.816 | **0.605** | 0.761 | 0.605 |
| CNNDetection | 0.963 | 0.936 | 0.910 | **0.890** | 0.822 | 0.890 | 0.822 |
| ResNet50 | 0.890 | 0.861 | 0.838 | 0.724 | 0.696 | 0.780 | 0.696 |
| **LEDD (concat)** | **0.975** | **0.945** | **0.916** | 0.868 | **0.912** | **0.910** | **0.868** |

Wins 4 of 5 conditions, the degraded mean by **0.020**, and the worst case by **0.046**.
The only loss is blur, to CNNDetection — whose method *is* blur augmentation.

**Novel finding:** NPR collapses to **0.605 under resizing**, near chance. Its mechanism
reads neighbouring-pixel relationships left by up-sampling, which resampling destroys.
Tan et al. (CVPR 2024) report no robustness experiments; this supplies the missing
evaluation and explains the failure mechanistically.

---

## 4b. Out-of-distribution generators (post-GenImage)

Self-collected set. Fakes generated from **COCO val2017 captions** so content matches the
reals; reals are COCO val2017 images. 400 real + 400 fake per generator. Preprocessed with
**identical settings to the training archive** (224 centre-crop, `--equalize-jpeg 0`).
Archive note: 3 of 800 images (0.4%) had a short side under 224 and were upscaled.

Two OOD generators, deliberately from **different architecture families**:
**SDXL** (UNet latent diffusion) and **PixArt-Sigma** (diffusion transformer). Both
postdate GenImage. SD3 was dropped (gated repo, access not granted); Flux was dropped
(12B parameters, 3-6 h of generation on a T4 for a third point that adds little).

| Generator | n | Blockiness floor | LEDD matched-compression | LEDD raw | Status |
|---|---|---|---|---|---|
| **SDXL** | 800 | 0.582 (mild, dev 0.082) | **0.922** | **0.983** | done |
| **PixArt-Sigma** (equalised q95) | 800 | 0.444 (mild, dev 0.056) | — | **0.985** | done |

Note: equalisation was applied to PixArt-Sigma only. Applied to SDXL it made the bias
*worse* (0.582 -> 0.379, dev 0.082 -> 0.121), because re-encoding a pristine high-detail
render introduces more blocking than re-encoding an already-compressed photograph. We
therefore use the original SDXL set and the equalised PixArt-Sigma set, reporting the
blockiness floor for each. **Equalisation is not universally beneficial and should itself
be audited.**

### SDXL — all models, same set

| Model | Params | Clean | JPEG-75 | **JPEG-50 (matched)** | Blur | Resize | Deg. mean | Worst |
|---|---|---|---|---|---|---|---|---|
| NPR | 1.44M | 0.848 | 0.793 | 0.768 | 0.814 | **0.588** | 0.741 | 0.588 |
| ResNet50 | 25.6M | 0.909 | 0.861 | 0.823 | 0.753 | 0.778 | 0.804 | 0.753 |
| CNNDetection | 25.6M | 0.941 | 0.919 | 0.897 | **0.879** | 0.845 | 0.885 | 0.845 |
| **LEDD** | **5.6M** | **0.983** | **0.950** | **0.922** | 0.845 | **0.912** | **0.907** | 0.845 |

**LEDD leads on the compression-matched figure by +0.025** (0.922 vs 0.897) and on clean by
+0.042. Degraded mean +0.022. Worst case ties CNNDetection at 0.845, but the failure modes
differ: LEDD's weak point is blur, CNNDetection's is resize.

**JPEG-50 is the number for the abstract** — both classes compressed, so the COCO-JPEG /
PNG-fake asymmetry is largely neutralised.

**NPR's resize collapse replicates**: 0.588 here, 0.605 on Midjourney. Two independent
datasets, same mechanism (resampling destroys the neighbouring-pixel relationships NPR
depends on). Tan et al. report no robustness experiments.

### PixArt-Sigma — all models, same set

Equalised set (blockiness 0.444, dev 0.056). These are final.

| Model | Params | Clean | JPEG-75 | JPEG-50 | Blur | Resize | Deg. mean | Worst |
|---|---|---|---|---|---|---|---|---|
| NPR | 1.44M | 0.777 | 0.811 | 0.781 | 0.704 | 0.665 | 0.740 | 0.665 |
| ResNet50 | 25.6M | 0.912 | 0.892 | 0.855 | 0.753 | 0.817 | 0.829 | 0.753 |
| CNNDetection | 25.6M | **0.991** | **0.982** | **0.966** | **0.903** | 0.922 | 0.943 | **0.903** |
| **LEDD** | **5.6M** | 0.985 | 0.973 | 0.945 | 0.901 | **0.962** | **0.946** | 0.901 |

**LEDD and CNNDetection are statistically indistinguishable on PixArt-Sigma.** Degraded
mean 0.9455 vs 0.9432 (delta +0.0023 to LEDD); worst case 0.9012 vs 0.9028 (delta -0.0016).
CNNDetection leads on clean and JPEG by 0.006-0.020; LEDD leads on resize by 0.040.

**Equalisation changed this conclusion.** On the un-equalised set CNNDetection appeared to
win the degraded mean by 0.005; the compression bias was inflating its blur score
(0.939 -> 0.903 after equalisation) and NPR's more severely (0.802 -> 0.704). Report the
equalised numbers only.

Across both OOD generators: **LEDD leads on SDXL, ties CNNDetection on PixArt-Sigma.**

NPR's resize collapse appears a **third** time: 0.665, its worst column again.

### The raw 0.983 is optimistic — report both numbers

COCO reals are **JPEG**; SDXL fakes are **PNG that have never been compressed**. That is
the real-vs-fake compression asymmetry in its purest form, and this model is known to be
sensitive to it (band 11 carried 0.27 AUC in the frequency stream).

**The JPEG-50 row is the partial control.** With both classes compressed, the asymmetry is
largely neutralised and AUC is still **0.922** — if the model were merely detecting "has
JPEG history" that row would fall toward 0.5. So there is genuine generator signal, but the
clean figure is inflated.

**Measured floor:** the blockiness audit on this set gives **AUC 0.582 (deviation 0.082,
"mild")** — i.e. a classifier using only an 8x8 blockiness statistic reaches 0.582 where
LEDD reaches 0.983. For comparison the Midjourney test generator inside GenImage is worse
at 0.668, so this self-collected set is *cleaner* than the benchmark's own hardest split.

**Report all three: floor 0.582, matched-compression 0.922, raw 0.983.** That brackets the
confound with measurements instead of assertion.

**SUPERSEDED (in progress):** both OOD sets are being re-encoded at q95 so that reals and
fakes share one compression history. PixArt-Sigma already drops from 0.653 to **0.444
(dev 0.056)** after equalisation. Once both sets are re-evaluated, report the equalised
numbers as primary and this bracketing discussion becomes a methods footnote.

Context: SDXL postdates GenImage by roughly two years, and FIRE reports DIRE collapsing to
about 0.50 AUC on SDXL-era images.

---

## 5. Ablations

All at 10 epochs, **single seed (42)** — unlike the seeded main table in S2. Mean =
average over all epochs, which is the honest measure: best-epoch selection rewards
whichever run was noisiest.

**Read the last three rows against cross-attention, not against concatenation.** All three
inherit `stage3_joint.yaml`, whose fusion mode is `cross_attention`; they were run before
concatenation was adopted. Their base is therefore the 0.905 row, not the 0.928 row.

| Setting (seed 42) | Best | Mean | Swing | vs its base | Verdict |
|---|---|---|---|---|---|
| Cross-attention fusion (base for rows 3–5) | 0.946 | 0.905 | 0.097 | — | — |
| **Concatenation** | 0.937 | **0.928** | 0.017 | +0.023 vs cross-attn | **simpler wins** |
| Cross-attn − modality dropout | 0.929 | 0.903 | 0.053 | −0.002 | inside the swing |
| Cross-attn − contrastive loss | 0.934 | 0.886 | 0.298 | −0.019 | swing 0.298 — uninterpretable |
| Cross-attn − band-entropy | 0.941 | **0.913** | 0.091 | +0.008 | inside the swing |

**Correction to an earlier reading.** "Four of four added components fail to help" is not what
these rows show. Removing the contrastive loss *lowered* the mean by 0.019, which on its face
suggests it helps; that run's within-run swing is 0.298, so the number means nothing at one
seed. The defensible claim is the weaker one: **no configuration carrying these components
reaches concatenation's 0.928, so none was adopted** — not that each was individually useless.
Every between-config difference here (0.002–0.023) is smaller than the within-run swings
(0.017–0.298).

Cross-check: with three seeds (S2) the fusion comparison is concat 0.923 ± 0.005 vs
cross-attention 0.906 ± 0.001, non-overlapping. The single-seed ablations above are
directional only.

**Frequency representation** (frequency stream alone, last-5-epoch mean):
magnitude 0.820 · **DCT 0.822** · magnitude+phase 0.810.
Magnitude and DCT equivalent; **adding phase hurts**.

**Band count** (same measure): 8 → 0.814 · **12 → 0.820** · 16 → 0.804.
Twelve confirmed optimal — too few loses resolution, too many dilutes each band.

Two design choices are validated by data. For the components, see the correction above.

---

## 6. Explanation faithfulness

Deletion: lower is better. Insertion: higher is better. Random-saliency control included.

| Explanation | Result | Verdict |
|---|---|---|
| Spatial maps — insertion | 0.765 vs random 0.566 | **faithful** |
| Spatial maps — deletion | 0.625 vs random 0.577 | fails; a masked patch is itself smooth, which a synthetic-image detector reacts to |
| Stream balance metric | claimed 97% frequency vs **causal 4%** | **unfaithful — report causal only** |

### Attention is not explanation (three demonstrations)

1. **Band level, frequency stream alone.** Attention is mid-band: it **peaks at rings 4–5**
   (radius 0.58–0.71) and **56% of its mass sits on rings 3–6**, for real images. Peak and
   mass are two different statements — do not merge them, as an earlier revision of this
   ledger did. Both match FIRE's published claim. Causal band deletion disagrees:
   **band 11 (radius 0.96–1.00) costs 0.270 AUC** (0.830 → 0.560) and band 1 costs 0.156,
   each measured independently, not cumulatively, while the mid-band rings cost ~0.00.
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
MobileViT backbone already encodes it at inference time.

### ...but it does help during training

Matched control — spatial-only, **same initialisation (Stage 1 checkpoint) and same
schedule (lr_backbone 1e-5, 10 epochs)** as the concat model:

Both sides, 3 seeds:

| | best | mean | last-5 |
|---|---|---|---|
| Spatial only (matched init + schedule) | 0.9321 ± 0.0042 | 0.9083 ± 0.0032 | 0.9170 ± 0.0042 |
| Concat (dual-stream) | 0.9350 ± 0.0027 | **0.9230 ± 0.0046** | 0.9301 ± 0.0021 |
| **delta** | +0.0028 | **+0.0146 (3.7 sigma)** | +0.0131 |

Note the **best-epoch delta is only +0.003**. Reporting best-epoch — the field norm —
would have shown no difference at all. Third independent argument for the mean.

**The two results are not in conflict — they measure different things.** Training *with*
the frequency stream yields a better model (**+0.0146 mean, 3 seeds each**), while removing
that stream from the *trained* model costs 0.003. The second stream therefore behaves as an **auxiliary
training signal rather than a source of complementary inference-time evidence**.
Candidate mechanisms (untested): regularisation of the shared head, extra gradient signal
shaping the spatial features, or added capacity.

This nuance is invisible to either experiment alone, and no prior work runs both.
Earlier framing — "frequency modelling is simply redundant" — is **too strong**; it is
redundant at inference, not during training.

Control seeds: 0.9323 / 0.9279 / 0.9362 (best), means 0.9047 / 0.9096 / 0.9107.

Consequence for the paper: report band attribution for the **standalone frequency stream**
(a <1M-parameter, 0.829-AUC detector where band 11 causally matters), not for the joint
model.

---

## 7. Efficiency

**CORRECTED 2026-10-23.** The figure 6,306,633 that appeared here and throughout earlier
drafts is the **cross-attention** model — `measure_efficiency.py` was run with
`configs/kaggle.yaml`, which inherits `default.yaml` and its `mode: cross_attention`. The
reported model is concatenation, and it is smaller:

| Model | Total params | Deployed (no projection head) | Fusion block |
|---|---|---|---|
| **Concat (reported)** | **5,637,321** | **5,575,561** | 130,752 |
| Cross-attention (rejected) | 6,306,633 | 6,244,873 | 800,064 |

The whole 669,312 difference is the fusion block. Report **5.6M**, not 6.3M. This makes the
efficiency claim stronger, not weaker: 5.64M against 25.6M is under a quarter.

Also note `export_model.py` originally summed `state_dict()` entries, which include
BatchNorm buffers (12,288), and so printed 5,587,849 rather than 5,575,561. Fixed; the
bundle now reports parameters and buffers separately.

5,637,321 parameters. No diffusion model and no large frozen encoder at inference.
FLOPs via fvcore **undercount** — `scaled_dot_product_attention` and `fft_fft2` are
unsupported operators, so report as "conv+linear FLOPs" or cross-check with thop.

---

## 8. Training behaviour worth reporting

**Cross-generator accuracy peaks early then declines.** The spatial stream reaches its
best ADM score at **epoch 1** (0.9167) and falls afterwards while in-distribution AUC
climbs to 0.9998.

In-distribution AUC ties at its 0.9998 maximum across epochs 7-11, where ADM ranges
0.8921-0.9039. Selecting on in-distribution validation would therefore have cost
**0.013 to 0.025 AUC depending on how the tie is broken** (first-tie 0.0128, last-tie
0.0246). Quote the range, not a single number. Direct evidence that the dedicated
validation generator was necessary.

**SupCon dominates the objective.** At λ=0.5 the contributions are BCE 0.17 vs SupCon
4.33 — roughly 1:25, not the configured 1:0.5. Report the effective ratio.

---

## 9. Open items

| Item | Status |
|---|---|
| `ctrl_spatial_from_s1` — spatial-only, matched, 3 seeds | **done**: mean 0.9083 ± 0.0032 |
| CNNDetection / ResNet50 re-run with fixed optimiser | **done** |
| Baseline seeds (2 more each) | optional; baselines are single-seed, LEDD has three |
| OOD set — SDXL | **done**: 0.983 raw / 0.922 matched-compression |
| OOD set — PixArt-Sigma | **done** (equalised, dev 0.056) |
| SD3 | dropped — gated repo, access not granted |
| Flux | dropped — 12B params, 3-6 h generation for marginal gain |
| Equalise OOD sets at q95 | **done**. PixArt 0.653 -> 0.444 (improved). SDXL 0.582 -> 0.379 (worse) — original SDXL set retained |
| Baselines evaluated on the SDXL set | **done** |
| Baselines on PixArt-Sigma | **done** (equalised) |
| FIRE baseline | pending (6–10 h), optional — may cite published numbers instead |

---

## 10. Claims the paper can make

1. A 5.6M-parameter detector **beats all 25M baselines on the held-out test generator**
   (0.975 vs 0.963 / 0.890) and **on degraded images** (mean 0.910 vs 0.890; worst case
   0.868 vs 0.822).
2. **Attention-based explanations are unfaithful**, shown three independent ways.
3. **The frequency stream helps during training but is unused at inference**:
   **+0.0146 ± 0.004 (3.7 sigma, 3 seeds each)** when trained with it, but **−0.003** when
   removed from the trained model. It behaves as an auxiliary training signal, not as
   complementary evidence. The best-epoch delta is only +0.003, so this is invisible
   under the field's usual reporting convention.
4. **NPR fails under resizing** — 0.605 on Midjourney, 0.588 on SDXL and 0.665 on
   PixArt-Sigma: three independent datasets. An unreported weakness of a CVPR 2024 method,
   with a mechanistic explanation.
5. **Generalises to generators released after the benchmark**, tested on two architecture
   families: SDXL (UNet latent diffusion) and PixArt-Sigma (diffusion transformer).
   **LEDD leads all baselines on SDXL (+0.042 clean, +0.025 matched-compression) and ties
   CNNDetection on PixArt-Sigma** (degraded mean 0.9455 vs 0.9432).
6. **Compression equalisation is not universally beneficial.** It halved the bias on
   PixArt-Sigma (0.653 -> 0.444) but worsened it on SDXL (0.582 -> 0.379), and it changed
   which model appeared to win PixArt-Sigma. Benchmark builders should audit the
   equalisation itself.
7. **Best-epoch selection on a single validation generator can pick an undertrained
   model.** ResNet50's best ADM epoch was epoch 0; it then lost the test generator by
   0.085. A limitation of the protocol used here, reported rather than hidden.
8. Evaluation controls that are not standard in this literature: a spectral leak check, a
   per-generator compression-bias audit, a three-role generator split, seeded ablations
   reported by mean rather than best epoch, and threshold calibration fitted off the test
   split.

## Claims the paper cannot make

- That cross-attention fusion beats concatenation. It does not.
- That the contrastive loss, band-entropy regulariser or modality dropout help. No
  configuration using them beats plain concatenation. The single-seed rows in S5 do **not**
  support the stronger claim that each is individually useless — the contrastive-loss row
  points the other way on a run that is too unstable to read.
- That the frequency stream supplies complementary evidence at inference. It does not
  (0.003).
- That the frequency stream improves best-epoch performance. It does not (+0.003, within noise).
- That SDXL AUC is 0.983 without qualification — quote it alongside the 0.582 blockiness
  floor and the 0.922 matched-compression figure.
- That LEDD beats all baselines on out-of-distribution generators. It wins SDXL and ties
  CNNDetection on PixArt-Sigma (and loses PixArt-Sigma on clean and JPEG specifically).
