"""End-to-end model tests. These need timm for MobileViT; skipped if unavailable."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from ledd.losses import CombinedLoss
from ledd.models import LEDDDetector, count_parameters


@pytest.fixture
def model(tiny_cfg):
    return LEDDDetector(tiny_cfg)


def test_forward_produces_all_explainability_outputs(model):
    from ledd.models.attention import set_attention_recording

    set_attention_recording(model, True)
    out = model(torch.rand(2, 3, 64, 64), return_embeddings=True)
    assert out["logit"].shape == (2,)
    assert out["proj"].shape == (2, 16)
    assert out["band_attention"].shape == (2, 8)
    assert out["balance"].shape == (2, 2)


def test_projection_is_l2_normalised(model):
    p = model(torch.rand(2, 3, 64, 64), return_embeddings=True)["proj"]
    assert torch.allclose(p.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_band_and_spatial_masks_change_the_prediction(model):
    x = torch.rand(2, 3, 64, 64)
    model.eval()
    base = model(x)["logit"]
    no_bands = model(x, band_mask=torch.zeros(2, 8))["logit"]
    gh, gw = model(x)["grid"]
    no_spatial = model(x, spatial_token_mask=torch.zeros(2, gh * gw))["logit"]
    assert not torch.allclose(base, no_bands, atol=1e-4)
    assert not torch.allclose(base, no_spatial, atol=1e-4)


def test_full_model_stays_lightweight():
    """The headline claim: a full-size model must land near ~7M params."""
    cfg = {
        "data": {"image_size": 224},
        "model": {
            "spatial": {"name": "mobilevit_s", "pretrained": False},
            "frequency": {"n_bands": 12, "embed_dim": 160, "depth": 2, "heads": 4},
            "fusion": {"dim": 192, "heads": 4, "layers": 1, "bidirectional": True},
            "head": {"hidden": 192, "dropout": 0.1},
            "projection": {"dim": 128, "hidden": 192},
        },
    }
    n = count_parameters(LEDDDetector(cfg))
    assert n < 9_000_000, f"model has {n} params -- lightweight claim at risk"


def test_param_groups_separate_backbone_from_new_parts(model):
    groups = model.param_groups(1e-5, 1e-3)
    assert len(groups) == 3
    assert groups[0]["lr"] == 1e-5 and groups[1]["lr"] == 1e-3
    assert groups[2]["weight_decay"] == 0.0        # norms/biases excluded from decay


def test_combined_loss_backward(model, tiny_cfg):
    from ledd.models.attention import set_attention_recording

    set_attention_recording(model, True)
    crit = CombinedLoss(tiny_cfg)
    out = model(torch.rand(4, 3, 64, 64), return_embeddings=True)
    losses = crit(out, torch.tensor([0.0, 1.0, 0.0, 1.0]))
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert {"bce", "supcon", "band_entropy"} <= set(losses)
