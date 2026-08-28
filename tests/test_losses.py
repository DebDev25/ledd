import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from ledd.losses import FeatureQueue, band_entropy_loss, supcon_loss
from ledd.losses.supcon import SupConCriterion


def _norm(x):
    return F.normalize(x, dim=-1)


def test_supcon_lower_when_classes_are_separated():
    """The loss must reward exactly what we want: reals together, fakes together."""
    torch.manual_seed(0)
    d = 16
    a, b = _norm(torch.randn(1, d)), _norm(torch.randn(1, d))
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    tight = _norm(torch.cat([a.repeat(3, 1), b.repeat(3, 1)]) + 0.01 * torch.randn(6, d))
    scattered = _norm(torch.randn(6, d))

    assert supcon_loss(tight, labels) < supcon_loss(scattered, labels)


def test_supcon_ignores_generator_identity():
    """Two fakes from different generators are positives for each other -- that is
    the entire cross-generator mechanism."""
    torch.manual_seed(0)
    d = 8
    fake_dir = _norm(torch.randn(1, d))
    feats = _norm(torch.cat([
        _norm(torch.randn(2, d)),                       # reals
        fake_dir.repeat(2, 1) + 0.01 * torch.randn(2, d),  # fakes, "two generators"
    ]))
    labels = torch.tensor([0, 0, 1, 1])
    loss_same_class_together = supcon_loss(feats, labels)

    swapped = feats.clone()[[0, 2, 1, 3]]               # break the class grouping
    assert loss_same_class_together < supcon_loss(swapped, labels)


def test_supcon_is_finite_with_queue():
    feats, labels = _norm(torch.randn(8, 16)), torch.randint(0, 2, (8,))
    qf, ql = _norm(torch.randn(64, 16)), torch.randint(0, 2, (64,))
    v = supcon_loss(feats, labels, qf, ql)
    assert torch.isfinite(v) and v > 0


def test_supcon_handles_single_class_batch():
    """Must not NaN if a batch happens to be all-real or all-fake."""
    feats, labels = _norm(torch.randn(4, 16)), torch.zeros(4, dtype=torch.long)
    assert torch.isfinite(supcon_loss(feats, labels))


def test_queue_warmup_disables_loss_then_ramps():
    c = SupConCriterion(dim=16, queue_size=32, warmup_steps=3, ramp_steps=2)
    c.train()
    feats, labels = _norm(torch.randn(4, 16)), torch.randint(0, 2, (4,))
    assert c(feats, labels).item() == 0.0           # warmup: loss off, queue filling
    for _ in range(3):
        c(feats, labels)
    assert c.weight_scale() > 0                     # ramp has started


def test_feature_queue_wraps_around():
    q = FeatureQueue(dim=8, size=10)
    for _ in range(4):
        q.enqueue(_norm(torch.randn(4, 8)), torch.randint(0, 2, (4,)))
    assert len(q) == 10
    f, l = q.get()
    assert f.shape == (10, 8) and l.shape == (10,)


def test_band_entropy_penalises_collapse():
    n = 8
    uniform = torch.full((1, n), 1.0 / n)
    collapsed = torch.zeros(1, n)
    collapsed[0, 3] = 1.0
    assert band_entropy_loss(uniform) < band_entropy_loss(collapsed)
    assert band_entropy_loss(uniform).item() == pytest.approx(0.0, abs=1e-5)
    assert band_entropy_loss(collapsed).item() == pytest.approx(1.0, abs=1e-3)
