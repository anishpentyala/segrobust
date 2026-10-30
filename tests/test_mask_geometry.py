import numpy as np
import pytest

from segrobust import GEOMETRIC, CORRUPTION_NAMES, corrupt_pair


def sample(h=48, w=48, seed=1):
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    mask = rng.randint(0, 3, size=(h, w), dtype=np.int64)
    return img, mask


@pytest.mark.parametrize('name', [n for n in CORRUPTION_NAMES
                                  if n not in GEOMETRIC])
def test_photometric_corruptions_leave_mask_untouched(name):
    img, mask = sample()
    out_img, out_mask = corrupt_pair(img, mask, name, severity=3, seed=3)
    assert np.array_equal(out_mask, mask)
    assert out_img.shape[:2] == mask.shape


def test_elastic_warps_mask_with_same_field():
    img, mask = sample()
    out_img, out_mask = corrupt_pair(img, mask, 'elastic_transform',
                                     severity=3, seed=11)
    assert out_mask.shape == mask.shape
    assert out_mask.dtype == mask.dtype
    # a geometric warp at severity >= 2 must actually move some labels
    assert not np.array_equal(out_mask, mask)
    # warping is nearest-neighbor: no new label values may appear
    assert set(np.unique(out_mask)) <= set(np.unique(mask))


def test_elastic_pair_is_deterministic():
    img, mask = sample()
    a = corrupt_pair(img, mask, 'elastic_transform', severity=4, seed=9)
    b = corrupt_pair(img, mask, 'elastic_transform', severity=4, seed=9)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_mask_warp_matches_image_warp():
    # The mask must be warped by the EXACT field applied to the image:
    # reproduce both from the same seed and compare against direct
    # application of the field.
    from segrobust.corruptions import elastic_fields, apply_elastic

    img, mask = sample()
    severity, seed = 3, 17
    out_img, out_mask = corrupt_pair(img, mask, 'elastic_transform',
                                     severity=severity, seed=seed)

    np.random.seed(seed)
    dx, dy = elastic_fields(img.shape, severity)
    exp_img = np.uint8(np.clip(
        apply_elastic(img.astype(np.float32) / 255., dx, dy, order=1),
        0, 1) * 255)
    exp_mask = apply_elastic(mask.astype(np.float32), dx, dy,
                             order=0).astype(mask.dtype)

    assert np.array_equal(out_img, exp_img), 'image not warped with field'
    assert np.array_equal(out_mask, exp_mask), \
        'mask warped with a different field'
