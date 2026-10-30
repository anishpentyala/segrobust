"""Public API: corrupt images deterministically, with mask geometry care."""

from contextlib import contextmanager

import numpy as np

from . import corruptions as _c

# category -> ordered corruption names (Hendrycks & Dietterich grouping)
_CATEGORIES = {
    'noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
    'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
    'weather': ['snow', 'frost', 'fog', 'brightness'],
    'digital': ['contrast', 'elastic_transform', 'pixelate',
                'jpeg_compression'],
}
_VALIDATION = ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']

# Corruptions that move image content geometrically. For these, a label
# mask must be warped with the identical displacement; everything else is
# photometric and leaves the mask unchanged (ImageNet-C convention).
GEOMETRIC = frozenset({'elastic_transform'})


def _registry():
    reg = {}
    for cat, names in _CATEGORIES.items():
        for n in names:
            reg[n] = {'fn': getattr(_c, n), 'category': cat,
                      'geometric': n in GEOMETRIC, 'validation': False}
    for n in _VALIDATION:
        reg[n] = {'fn': getattr(_c, n), 'category': 'validation',
                  'geometric': n in GEOMETRIC, 'validation': True}
    return reg


CORRUPTIONS = _registry()
CORRUPTION_NAMES = list(CORRUPTIONS)


def get_corruption_names(subset='common'):
    """Names for 'common' (first 15), 'validation' (last 4), 'all', or a
    category ('noise'/'blur'/'weather'/'digital')."""
    if subset == 'common':
        return [n for cat in _CATEGORIES.values() for n in cat][:15]
    if subset == 'validation':
        return list(_VALIDATION)
    if subset == 'all':
        return CORRUPTION_NAMES
    if subset in _CATEGORIES:
        return list(_CATEGORIES[subset])
    raise ValueError(
        "subset must be one of 'common', 'validation', 'all', "
        "'noise', 'blur', 'weather', 'digital'")


@contextmanager
def _seeded(seed):
    """Seed the legacy global RNG for the duration of one corruption call.

    The vendored corruption functions draw from np.random's global state.
    We save and restore that state so a seeded call is reproducible AND
    does not disturb the caller's RNG stream.
    """
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _validate_image(image):
    if not isinstance(image, np.ndarray):
        raise TypeError('image must be a numpy.ndarray')
    if image.dtype != np.uint8:
        raise TypeError('image dtype must be np.uint8')
    if image.ndim not in (2, 3):
        raise ValueError('image shape must be (H, W) or (H, W, C)')
    if image.ndim == 3 and image.shape[2] not in (1, 3):
        raise ValueError('image channels (last dim) must be 1 or 3')
    if min(image.shape[:2]) < 8:
        raise ValueError('image height and width must be at least 8 pixels')


def _to_rgb(image):
    if image.ndim == 2:
        return np.stack((image,) * 3, axis=-1)
    if image.shape[2] == 1:
        return np.concatenate([image] * 3, axis=-1)
    return image


def corrupt(image, corruption_name, severity=1, seed=None):
    """Return a corrupted copy of ``image``.

    image: uint8 ndarray (H, W) or (H, W, 1|3), values in [0, 255].
    severity: int in [1, 5].
    seed: optional int. Given a seed, the result is a deterministic
        function of (image bytes, name, severity, seed), and the caller's
        np.random state is left untouched. Without a seed the global RNG
        is used directly.
    Returns uint8 ndarray with the same spatial shape as the input.
    """
    _validate_image(image)
    if severity not in (1, 2, 3, 4, 5):
        raise ValueError('severity must be an integer in [1, 5]')
    try:
        entry = CORRUPTIONS[corruption_name]
    except KeyError:
        raise ValueError(
            f'unknown corruption {corruption_name!r}; '
            f'choose from {CORRUPTION_NAMES}') from None

    was_single = image.ndim == 2 or image.shape[2] == 1
    rgb = _to_rgb(image)
    with _seeded(seed):
        out = entry['fn'](rgb, severity)
    out = np.uint8(np.clip(np.array(out), 0, 255))
    if was_single and out.ndim == 3:
        # single-channel input returns single-channel output
        gray = _c.rgb2gray(out) if out.shape[2] == 3 else out[..., 0]
        out = np.uint8(np.clip(gray, 0, 255))
        if image.ndim == 3:
            out = out[..., np.newaxis]
    return out


def corrupt_pair(image, mask, corruption_name, severity=1, seed=None,
                 ignore_index=None):
    """Corrupt ``image`` and keep ``mask`` (int labels, HxW) aligned.

    Photometric corruptions leave the mask untouched. Geometric
    corruptions (currently only elastic_transform) warp the mask with the
    identical displacement field using nearest-neighbor sampling, so label
    boundaries track the warped image content. ``ignore_index`` label
    values are preserved by nearest sampling like any other label.
    """
    if not isinstance(mask, np.ndarray) or mask.shape[:2] != image.shape[:2]:
        raise ValueError('mask must be an ndarray with the same H, W as image')
    if corruption_name in GEOMETRIC:
        _validate_image(image)
        if severity not in (1, 2, 3, 4, 5):
            raise ValueError('severity must be an integer in [1, 5]')
        rgb = _to_rgb(image)
        with _seeded(seed):
            dx, dy = _c.elastic_fields(rgb.shape, severity)
            warped_img = np.clip(
                _c.apply_elastic(rgb.astype(np.float32) / 255., dx, dy,
                                 order=1), 0, 1) * 255
            warped_mask = _c.apply_elastic(
                mask.astype(np.float32), dx, dy, order=0)
        warped_mask = warped_mask.astype(mask.dtype)
        was_single = image.ndim == 2 or image.shape[2] == 1
        if was_single:
            gray = np.uint8(np.clip(_c.rgb2gray(np.uint8(warped_img)), 0, 255))
            warped_img = gray[..., np.newaxis] if image.ndim == 3 else gray
        else:
            warped_img = np.uint8(warped_img)
        return warped_img, warped_mask
    return corrupt(image, corruption_name, severity, seed), mask.copy()
