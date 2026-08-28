import pytest

torch = pytest.importorskip("torch")

from ledd.models.attention import set_attention_recording
from ledd.models.fusion import FusionModule


def make(mode="cross_attention", **kw):
    args = dict(spatial_dim=64, freq_dim=32, dim=32, heads=4, layers=1,
                bidirectional=True, modality_dropout=0.0, mode=mode)
    args.update(kw)
    return FusionModule(**args)


def test_cross_attention_output_shape():
    f = make()
    set_attention_recording(f, True)
    out = f(torch.randn(2, 196, 64), torch.randn(2, 12, 32))
    assert out["fused"].shape == (2, 32)


def test_balance_is_token_count_normalised():
    """196 spatial vs 12 band tokens would fake a spatial win if we summed raw
    attention mass. Uniform attention must give a ~50/50 balance."""
    f = make()
    set_attention_recording(f, True)
    out = f(torch.randn(4, 196, 64), torch.randn(4, 12, 32))
    b = out["balance"]
    assert b.shape == (4, 2)
    assert torch.allclose(b.sum(dim=1), torch.ones(4), atol=1e-4)
    # untrained attention is near-uniform -> normalised balance near 0.5
    assert 0.3 < b.mean(dim=0)[0].item() < 0.7


def test_concat_mode_produces_no_balance():
    """The concat ablation is deliberately explanation-free; the harness must not
    silently report a meaningless balance for it."""
    out = make(mode="concat")(torch.randn(2, 196, 64), torch.randn(2, 12, 32))
    assert out["balance"] is None
    assert out["fused"].shape == (2, 32)


def test_modality_dropout_only_in_training():
    f = make(modality_dropout=1.0)
    s, q = torch.randn(8, 20, 64), torch.randn(8, 12, 32)
    f.eval()
    assert f(s, q)["dropped"] is None
    f.train()
    d = f(s, q)["dropped"]
    assert d is not None and (d[0] + d[1]).max() <= 1.0     # never drops both


def test_gradients_reach_both_streams():
    f = make()
    s = torch.randn(2, 20, 64, requires_grad=True)
    q = torch.randn(2, 12, 32, requires_grad=True)
    f(s, q)["fused"].sum().backward()
    assert s.grad.abs().sum() > 0 and q.grad.abs().sum() > 0
