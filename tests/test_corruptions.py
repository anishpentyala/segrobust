import numpy as np
import pytest

from segrobust import CORRUPTION_NAMES, corrupt, get_corruption_names


def sample_image(h=64, w=64, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize('name', CORRUPTION_NAMES)
@pytest.mark.parametrize('severity', [1, 3, 5])
def test_all_corruptions_run_and_preserve_shape(name, severity):
    img = sample_image()
    out = corrupt(img, name, severity=severity, seed=1)
    assert out.dtype == np.uint8
    assert out.shape[:2] == img.shape[:2]


@pytest.mark.parametrize('size', [8, 16, 32])
def test_small_images_all_corruptions(size):
    # upstream imagecorruptions rejects or crashes below 32px (issues #13/#21)
    img = sample_image(h=size, w=size)
    for name in get_corruption_names('all'):
        for severity in (1, 5):
            out = corrupt(img, name, severity=severity, seed=2)
            assert out.shape[:2] == (size, size), (name, severity)
            assert out.dtype == np.uint8


def test_determinism_same_seed_identical_output():
    img = sample_image()
    for name in CORRUPTION_NAMES:
        a = corrupt(img, name, severity=3, seed=42)
        b = corrupt(img, name, severity=3, seed=42)
        assert np.array_equal(a, b), name


def test_different_seeds_differ_for_stochastic_corruptions():
    img = sample_image()
    # gaussian_noise is stochastic; seed must change the draw
    a = corrupt(img, 'gaussian_noise', severity=3, seed=1)
    b = corrupt(img, 'gaussian_noise', severity=3, seed=2)
    assert not np.array_equal(a, b)


def test_seeded_call_does_not_pollute_global_rng():
    img = sample_image()
    np.random.seed(123)
    expected = np.random.rand(4)
    np.random.seed(123)
    corrupt(img, 'gaussian_noise', severity=3, seed=99)
    after = np.random.rand(4)
    assert np.array_equal(expected, after)


def test_unseeded_call_uses_global_rng():
    img = sample_image()
    np.random.seed(7)
    a = corrupt(img, 'gaussian_noise', severity=2)
    np.random.seed(7)
    b = corrupt(img, 'gaussian_noise', severity=2)
    assert np.array_equal(a, b)


def test_grayscale_input_roundtrip():
    img = sample_image()[..., 0]
    out = corrupt(img, 'gaussian_blur', severity=2, seed=0)
    assert out.shape == img.shape
    out3 = corrupt(img[..., np.newaxis], 'gaussian_blur', severity=2, seed=0)
    assert out3.shape == img.shape + (1,)


def test_validation_rejects_bad_inputs():
    img = sample_image()
    with pytest.raises(ValueError):
        corrupt(img, 'not_a_corruption')
    with pytest.raises(ValueError):
        corrupt(img, 'gaussian_noise', severity=0)
    with pytest.raises(TypeError):
        corrupt(img.astype(np.float32), 'gaussian_noise')
    with pytest.raises(ValueError):
        corrupt(sample_image(4, 4), 'gaussian_noise')


def test_corruption_actually_changes_image():
    img = sample_image()
    for name in CORRUPTION_NAMES:
        out = corrupt(img, name, severity=5, seed=5)
        assert not np.array_equal(out, img), f'{name} left image unchanged'


def _reference_plasma_fractal(mapsize, wibbledecay=3):
    """Verbatim diamond-square from the ImageNet-C reference
    (Hendrycks & Dietterich, make_imagenet_c.py) / imagecorruptions 1.1.2,
    used to check the vendored implementation against its source."""
    import math as _math  # noqa: F401
    maparray = np.zeros((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        ms = maparray.shape[0]
        drgrid = maparray[stepsize // 2:ms:stepsize, stepsize // 2:ms:stepsize]
        ulgrid = maparray[0:ms:stepsize, 0:ms:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:ms:stepsize, stepsize // 2:ms:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:ms:stepsize, 0:ms:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def test_plasma_fractal_matches_reference_algorithm():
    """The vendored plasma_fractal must match the diamond-square reference
    it vendors (regression test: the initial vendoring dropped both
    ``ulgrid +`` terms and wrote the left-diamond means onto the corner
    grid)."""
    from segrobust.corruptions import plasma_fractal
    for mapsize in (8, 32, 64):
        np.random.seed(1234)
        got = plasma_fractal(mapsize=mapsize)
        np.random.seed(1234)
        want = _reference_plasma_fractal(mapsize)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
