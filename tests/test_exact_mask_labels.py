"""Elastic nearest-neighbor sampling must preserve integer label identity."""
import numpy as np
import pytest

from segrobust import corrupt_pair
from segrobust.corruptions import elastic_fields, apply_elastic


def image():
    return np.random.RandomState(5).randint(0, 256, (16, 24, 3), dtype=np.uint8)


@pytest.mark.parametrize('dtype,label', [
    (np.int32, 16777217), (np.int64, 2147483647),
    (np.int64, np.iinfo(np.int64).max),
    (np.int64, np.iinfo(np.int64).min + 1),
    (np.uint64, np.iinfo(np.uint64).max),
])
def test_constant_integer_label_is_never_changed(dtype, label):
    img = image()
    mask = np.full(img.shape[:2], label, dtype=dtype)
    _, actual = corrupt_pair(img, mask, 'elastic_transform', 3, seed=7)
    assert np.array_equal(actual, mask), 'nearest warp must preserve constant label bits'
    assert actual.dtype == mask.dtype


@pytest.mark.parametrize('dtype,labels', [
    (np.int32, [0, 16777217, 2147483647]),
    (np.int64, [0, 16777217, np.iinfo(np.int64).max]),
    (np.int64, [np.iinfo(np.int64).min + 1, -16777217, -1]),
    (np.uint64, [0, 2**53+1, np.iinfo(np.uint64).max]),
])
@pytest.mark.parametrize('severity', [1, 3, 5])
def test_nonconstant_labels_match_exact_nearest_coordinates(dtype, labels, severity):
    img = image()
    labels = np.array(labels, dtype=dtype)
    mask = labels[np.indices(img.shape[:2]).sum(axis=0) % len(labels)]
    before = mask.copy()
    out_img, actual = corrupt_pair(img, mask, 'elastic_transform', severity, seed=11)
    state = np.random.get_state()
    try:
        np.random.seed(11)
        dx, dy = elastic_fields(img.shape, severity)
    finally:
        np.random.set_state(state)
    # Independent row/column sampling, rather than converting class values.
    rows, cols = np.indices(mask.shape, dtype=np.float64)
    rr = apply_elastic(rows, dx, dy, order=0).astype(np.intp)
    cc = apply_elastic(cols, dx, dy, order=0).astype(np.intp)
    expected = mask[rr, cc]
    assert np.array_equal(actual, expected), 'class values must follow exact source pixels'
    assert actual.dtype == dtype
    assert set(actual.flat) <= set(mask.flat)
    assert np.array_equal(mask, before)
    assert out_img.shape == img.shape


def test_large_ignore_label_is_preserved_and_rng_restored():
    img = image()
    mask = np.full(img.shape[:2], 2**53+1, dtype=np.int64)
    np.random.seed(153)
    before = np.random.get_state()
    _, actual = corrupt_pair(img, mask, 'elastic_transform', 4, seed=11,
                             ignore_index=2**53+1)
    after = np.random.get_state()
    assert np.array_equal(actual, mask), 'ignore label must not be changed by float rounding'
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_large_constant_label_regression_control():
    img = image()
    mask = np.full(img.shape[:2], 16777217, dtype=np.int32)
    _, actual = corrupt_pair(img, mask, 'elastic_transform', 3, seed=7)
    assert np.array_equal(actual, mask), 'nearest mask warp must not invent label 16777216'


@pytest.mark.parametrize('view', ['fortran', 'reverse', 'transpose', 'stride'])
def test_noncontiguous_masks_keep_source_pixel_identity(view):
    img = image()
    source = np.arange(16*24, dtype=np.int64).reshape(16, 24) + 2**53 + 1
    if view == 'fortran':
        mask = np.asfortranarray(source)
    elif view == 'reverse':
        mask = source[::-1, ::-1]
    elif view == 'transpose':
        mask = source.T.copy().T
    else:
        mask = np.repeat(source, 2, axis=1)[:, ::2]
    expected_indices = np.arange(mask.size, dtype=np.int64).reshape(mask.shape)
    _, sampled_indices = corrupt_pair(img, expected_indices, 'elastic_transform',
                                      5, seed=13)
    _, actual = corrupt_pair(img, mask, 'elastic_transform', 5, seed=13)
    expected = mask.ravel()[sampled_indices]
    assert np.array_equal(actual, expected)


def test_unseeded_pair_uses_same_global_rng_draws_for_any_label_bits():
    img = image()
    low = np.indices(img.shape[:2]).sum(axis=0).astype(np.int64) % 3
    high = np.array([2**53+1, 2**53+3, np.iinfo(np.int64).max])[low]
    state = np.random.get_state()
    try:
        np.random.seed(31)
        lo_img, lo_mask = corrupt_pair(img, low, 'elastic_transform', 5)
        lo_state = np.random.get_state()
        np.random.seed(31)
        hi_img, hi_mask = corrupt_pair(img, high, 'elastic_transform', 5)
        hi_state = np.random.get_state()
    finally:
        np.random.set_state(state)
    assert np.array_equal(lo_img, hi_img)
    assert np.array_equal(hi_mask, np.array([2**53+1, 2**53+3,
                                           np.iinfo(np.int64).max])[lo_mask])
    assert lo_state[0] == hi_state[0] and np.array_equal(lo_state[1], hi_state[1])
    assert lo_state[2:] == hi_state[2:]


def test_photometric_large_labels_still_return_independent_copy():
    img = image()
    mask = np.full(img.shape[:2], np.iinfo(np.uint64).max, dtype=np.uint64)
    _, actual = corrupt_pair(img, mask, 'brightness', 3, seed=1)
    assert np.array_equal(actual, mask)
    assert not np.shares_memory(actual, mask)


@pytest.mark.parametrize('dtype', ['>i8', '<i8', '>u8', '<u8'])
def test_readonly_endian_labels_preserve_values_and_dtype(dtype):
    img = image()
    mask = (np.arange(16*24, dtype=np.uint64).reshape(16,24) + 2**53 + 1).astype(dtype)
    mask.flags.writeable = False
    _, actual = corrupt_pair(img, mask, 'elastic_transform', 4, seed=17)
    state = np.random.get_state()
    try:
        np.random.seed(17)
        dx, dy = elastic_fields(img.shape, 4)
    finally:
        np.random.set_state(state)
    from scipy.ndimage import map_coordinates
    rows, cols = np.indices(mask.shape)
    coords = (rows + dy, cols + dx)
    # Sample low-bit offsets independently, then reconstruct original IDs.
    offsets = np.arange(mask.size, dtype=np.float64).reshape(mask.shape)
    expected_offsets = map_coordinates(offsets, coords, order=0, mode='reflect').astype(np.intp)
    assert np.array_equal(actual, mask.ravel()[expected_offsets])
    assert actual.dtype == mask.dtype
    assert not np.shares_memory(actual, mask)
