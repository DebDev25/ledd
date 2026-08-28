import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from ledd.explain import (deletion_insertion_bands, deletion_insertion_pixels,
                          explain_batch, normalize_map, random_baseline_maps,
                          stream_deletion_effect)
from ledd.models import LEDDDetector


@pytest.fixture
def model(tiny_cfg):
    m = LEDDDetector(tiny_cfg)
    m.eval()
    return m


def test_explain_batch_returns_all_maps(model):
    r = explain_batch(model, torch.rand(2, 3, 64, 64))
    assert r["spatial_map"].shape == (2, 64, 64)
    assert r["band_attribution"].shape == (2, 8)
    assert r["band_to_region"].shape == (2, 8, 64, 64)


def test_recording_is_disabled_after_explaining(model):
    """Attention buffers are large; leaving recording on would waste GPU memory
    for the rest of training."""
    from ledd.models.attention import RecordedAttention

    explain_batch(model, torch.rand(1, 3, 64, 64))
    assert all(not m.store_attn for m in model.modules() if isinstance(m, RecordedAttention))


def test_normalize_map_is_zero_one():
    m = normalize_map(torch.randn(3, 8, 8))
    assert m.min() >= 0 and m.max() <= 1


def test_deletion_insertion_pixels_returns_valid_aucs(model):
    x = torch.rand(2, 3, 64, 64)
    sal = normalize_map(explain_batch(model, x)["spatial_map"])
    r = deletion_insertion_pixels(model, x, sal, steps=8)
    assert 0 <= r["deletion_auc"] <= 1 and 0 <= r["insertion_auc"] <= 1
    assert r["fill"] == "mean"          # strategy must be recorded: AUCs depend on it
    assert len(r["deletion_curve"]) == 9


def test_band_deletion_curve_length_matches_bands(model):
    x = torch.rand(2, 3, 64, 64)
    attr = explain_batch(model, x)["band_attribution"]
    r = deletion_insertion_bands(model, x, attr)
    assert r["n_bands"] == 8
    assert len(r["deletion_curve"]) == 9


def test_random_control_exists(model):
    """Every faithfulness table needs the random-saliency control."""
    x = torch.rand(2, 3, 64, 64)
    r = deletion_insertion_pixels(model, x, random_baseline_maps((2, 64, 64)), steps=4)
    assert torch.isfinite(torch.tensor(r["deletion_auc"]))


def test_stream_deletion_produces_causal_balance(model):
    r = stream_deletion_effect(model, torch.rand(3, 3, 64, 64))
    assert r["causal_balance"].shape == (3, 2)
    assert torch.allclose(r["causal_balance"].sum(dim=1), torch.ones(3), atol=1e-4)
