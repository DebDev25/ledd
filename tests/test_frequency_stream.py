import pytest

torch = pytest.importorskip("torch")

from ledd.models.attention import set_attention_recording
from ledd.models.frequency import FrequencyClassifier, FrequencyStream


def make_stream(**kw):
    args = dict(n_bands=8, embed_dim=32, depth=2, heads=4, image_size=64)
    args.update(kw)
    return FrequencyStream(**args)


def test_forward_shapes():
    s = make_stream()
    out = s(torch.rand(3, 3, 64, 64))
    assert out["tokens"].shape == (3, 8, 32)
    assert out["cls"].shape == (3, 32)


def test_stream_is_lightweight():
    """The frequency stream must stay well under 1M params -- it is the cheap half
    of the 'lightweight' claim."""
    n = sum(p.numel() for p in FrequencyStream(n_bands=12, embed_dim=160, depth=2).parameters())
    assert n < 1_000_000, f"frequency stream has {n} params"


def test_band_attention_is_a_distribution():
    s = make_stream()
    set_attention_recording(s, True)
    s(torch.rand(2, 3, 64, 64))
    a = s.band_attention()
    assert a.shape == (2, 8)
    assert torch.allclose(a.sum(dim=1), torch.ones(2), atol=1e-5)


def test_band_mask_changes_output():
    """Band deletion must actually do something, or the faithfulness metric is fake."""
    s = FrequencyClassifier(make_stream())
    x = torch.rand(2, 3, 64, 64)
    full = s(x)["logit"]
    masked = s(x, band_mask=torch.zeros(2, 8))["logit"]
    assert not torch.allclose(full, masked, atol=1e-4)


def test_gradients_flow_to_band_encoder():
    s = FrequencyClassifier(make_stream())
    out = s(torch.rand(2, 3, 64, 64))
    out["logit"].sum().backward()
    g = s.stream.band_encoder.net[1].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
