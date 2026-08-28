import os

import numpy as np
import pytest
from PIL import Image

from ledd.data.degradations import (EVAL_DEGRADATIONS, RandomDegradation,
                                    apply_named, jpeg_compress, resize_down_up)
from ledd.data.prepare import center_crop_or_resize
from ledd.data.splits import SplitSpec


def _img(size=256):
    return Image.fromarray((np.random.rand(size, size, 3) * 255).astype("uint8"))


def test_center_crop_does_not_resize_large_images():
    """Crop-not-resize is a methodological requirement, not a preference:
    resampling imprints generator-correlated spectral signatures."""
    img, upscaled = center_crop_or_resize(_img(512), 224)
    assert img.size == (224, 224)
    assert upscaled is False


def test_small_images_are_flagged_as_upscaled():
    img, upscaled = center_crop_or_resize(_img(128), 224)
    assert img.size == (224, 224)
    assert upscaled is True         # must be reported in the paper if non-trivial


def test_all_named_degradations_run():
    img = _img(64)
    for name in EVAL_DEGRADATIONS:
        out = apply_named(img, name)
        assert out.size == img.size


def test_jpeg_actually_changes_pixels():
    img = _img(64)
    assert not np.array_equal(np.asarray(img), np.asarray(jpeg_compress(img, 30)))


def test_resize_down_up_preserves_size():
    assert resize_down_up(_img(64), 0.5).size == (64, 64)


def test_random_degradation_is_deterministic_under_seed():
    import random

    aug = RandomDegradation()
    img = _img(64)
    random.seed(0)
    a = np.asarray(aug(img))
    random.seed(0)
    b = np.asarray(aug(img))
    assert np.array_equal(a, b)


def test_split_spec_rejects_generator_in_two_roles():
    """A generator appearing in both train and test silently invalidates every
    generalisation number -- fail loudly instead."""
    spec = SplitSpec(train_generators=["a", "b"], val_generator="b", test_generators=["c"])
    with pytest.raises(ValueError, match="more than one role"):
        spec.validate()


def test_valid_split_spec_passes():
    SplitSpec(train_generators=["a", "b"], val_generator="c", test_generators=["d"]).validate()
