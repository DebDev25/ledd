"""The ring decomposition is the foundation of Stream 2 and of band attribution.
If these properties break, every frequency number in the paper is meaningless."""
import pytest

torch = pytest.importorskip("torch")

from ledd.models.fft import (band_pool, build_spectrum, equal_area_ring_masks,
                             log_magnitude_spectrum, radial_energy_profile, to_luma)


def test_rings_partition_the_disc():
    """Every pixel inside the inscribed disc belongs to exactly one ring; corner
    pixels and the DC bin belong to none."""
    m = equal_area_ring_masks(64, 64, n_bands=8, drop_lowest=0)
    total = m.sum(0)
    assert total.max() <= 1.0                     # no pixel in two rings
    from ledd.models.fft import radius_map

    inside = radius_map(64, 64) < 1.0
    inside[32, 32] = False                        # DC deliberately excluded
    assert torch.allclose(total[inside], torch.ones(int(inside.sum())))
    assert total[~inside].sum() == 0


@pytest.mark.parametrize("n", [8, 12, 16])
def test_rings_are_equal_area(n):
    """Equal-AREA is the whole point. Regression guard: normalising the radius by
    the CORNER distance instead of the inscribed radius made outer rings ~6x
    smaller than inner ones, biasing tokens and the attention distribution."""
    m = equal_area_ring_masks(256, 256, n_bands=n, drop_lowest=0)
    counts = m.sum(dim=(1, 2))
    assert (counts.max() / counts.min()).item() < 1.05


def test_dc_and_lowest_ring_are_dropped():
    m = equal_area_ring_masks(64, 64, n_bands=8, drop_lowest=1)
    assert m[0].sum() == 0                      # lowest ring zeroed
    assert m[:, 32, 32].sum() == 0              # DC bin zeroed everywhere


def test_band_pool_shapes_and_finiteness():
    spec = torch.randn(4, 1, 64, 64)
    masks = equal_area_ring_masks(64, 64, 8, drop_lowest=1)
    feats = band_pool(spec, masks, n_stats=4)
    assert feats.shape == (4, 8, 4)
    assert torch.isfinite(feats).all()


def test_radial_energy_profile_is_a_distribution():
    spec = torch.randn(3, 1, 64, 64)
    masks = equal_area_ring_masks(64, 64, 8, drop_lowest=1)
    p = radial_energy_profile(spec, masks)
    assert torch.allclose(p.sum(dim=1), torch.ones(3), atol=1e-5)


def test_spectrum_is_standardised():
    x = torch.rand(2, 3, 64, 64)
    s = log_magnitude_spectrum(to_luma(x))
    assert s.mean().abs() < 1e-4
    assert abs(s.std().item() - 1.0) < 0.1


@pytest.mark.parametrize("rep,ch", [("magnitude", 1), ("magnitude_phase", 2), ("dct", 1)])
def test_all_representations_build(rep, ch):
    """Covers the locked magnitude vs magnitude+phase vs DCT ablation."""
    s = build_spectrum(torch.rand(2, 3, 64, 64), rep)
    assert s.shape == (2, ch, 64, 64)
    assert torch.isfinite(s).all()


def test_spectrum_differs_between_smooth_and_noisy_images():
    """Sanity: the stream must see something discriminative.

    Compare the per-band PROFILE, never the spectrum's mean: the spectrum is
    standardised per image (zero mean, unit std) by design, so its mean is ~0 for
    every input and comparing means compares two zeros. That standardisation is
    deliberate -- it makes the frequency stream invariant to exposure and contrast.
    """
    torch.manual_seed(0)
    smooth = torch.ones(1, 3, 64, 64) * 0.5
    smooth += torch.linspace(0, 0.1, 64).view(1, 1, 1, -1)
    noisy = torch.rand(1, 3, 64, 64)

    masks = equal_area_ring_masks(64, 64, 8, drop_lowest=0)
    prof_s = band_pool(build_spectrum(smooth), masks, n_stats=1)[0, :, 0]
    prof_n = band_pool(build_spectrum(noisy), masks, n_stats=1)[0, :, 0]

    # a smooth image concentrates relatively more energy in the lowest ring
    assert (prof_s[0] - prof_n[0]).item() > 0.05
    # and the profiles differ overall (measured range across seeds: 0.17-0.33)
    assert torch.linalg.norm(prof_s - prof_n).item() > 0.10
