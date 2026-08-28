# What has and has not been verified

## Ran here (no GPU, no torch available in this environment)

* **Every file parses** (AST check across all 31 Python files).
* **13 tests pass**: `tests/test_config.py`, `tests/test_data.py` — config
  inheritance, CLI overrides, generator-role disjointness, crop-not-resize
  behaviour, all named degradations, augmentation determinism.
* **Core math re-derived in numpy** and checked independently of the torch code:
  ring partition/equal-area, band-pool denominators, radial-energy normalisation,
  SupCon L_out (including that it differs from L_in), band-entropy bounds,
  token-count-normalised balance, deletion/insertion AUC normalisation.
  14/16 passed on the first run — see below.

## Two bugs the math check caught (both fixed)

1. **Rings were not equal-area.** The radius was normalised by the *corner*
   distance, so every ring beyond r² > 0.5 was clipped by the square boundary —
   outer rings came out **~6.4x smaller** than inner ones. That would have biased
   the band tokens and, worse, the attention distribution the whole
   frequency-band attribution rests on. Fixed by normalising to the inscribed
   radius and excluding corner pixels; ring sizes now agree within 1–3%.
   `test_rings_are_equal_area` is a regression guard, parameterised over
   n_bands ∈ {8, 12, 16}.
2. **A test asserted a false property** — that every pixel belongs to exactly one
   ring, which the deliberately-zeroed DC bin violates. Rewritten to assert the
   true property (exact partition of the inscribed disc, nothing outside it).

## NOT verified — do this first on your machine

torch could not be installed in this sandbox, so nothing tensor-shaped has
actually executed. Before trusting anything:

```bash
pip install -r requirements.txt
pytest tests/ -q          # expect ~45 tests; all should pass on CPU in <2 min
```

Specifically unverified: MobileViT token-stage indexing against the real timm
feature list (the stride-16 lookup has a fallback, but the shapes have never
run), autocast/GradScaler interaction with the recorded-attention buffers,
Chefer relevance shapes through the fusion block, and the full training loop
end-to-end. Expect the first real bugs there.

Quickest smoke test once torch is in:

```python
import torch
from ledd.models import LEDDDetector, count_parameters
from ledd.utils.config import load_config
cfg = load_config("configs/stage3_joint.yaml")
m = LEDDDetector(cfg)
print(count_parameters(m))                      # expect ~7M
print(m(torch.rand(2, 3, 224, 224))["logit"].shape)
```
