"""The model's in-place preprocessing must not rewrite evaluation evidence."""
import numpy as np
import pytest

from segrobust import evaluate


@pytest.mark.parametrize('view', ['plain', 'noncontiguous', 'readonly'])
@pytest.mark.parametrize('classes', [False, True])
def test_model_gets_writable_copy_without_mutating_aliased_target(view, classes):
    image = np.ones((8, 16, 3), dtype=np.uint8)
    if view == 'noncontiguous':
        image = image[:, ::2]
    mask = image[..., 0]
    if view == 'readonly':
        image.flags.writeable = False
    before_image, before_mask = image.copy(), mask.copy()
    calls = []
    def model(im):
        assert im.flags.writeable
        assert not np.shares_memory(im, image)
        assert not np.shares_memory(im, mask)
        calls.append(im.shape)
        im[:] = 0
        return np.zeros(im.shape[:2], dtype=np.int64)
    result = evaluate(model, [image], [mask], 2, corruptions=[], severities=[1],
                      include_per_class=classes)
    assert calls == [image.shape]
    assert result['clean_miou'] == 0.0
    if classes:
        assert result['per_class']['clean']['target_pixels'] == [0, mask.size]
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(mask, before_mask)


@pytest.mark.parametrize('names', [['brightness'], ['contrast', 'brightness'], ['elastic_transform']])
@pytest.mark.parametrize('severities', [[1], [3, 1]])
def test_inplace_preprocessing_matches_explicit_copy_for_every_condition(names, severities):
    image = np.full((8, 8, 3), 180, dtype=np.uint8)
    mask = (np.indices((8, 8))[0] % 2).astype(np.int64)
    def pure(im):
        data = im.copy()
        data //= 2
        return (data[..., 0] > 80).astype(np.int64)
    seen = []
    def inplace(im):
        seen.append(im.copy())
        im //= 2
        return (im[..., 0] > 80).astype(np.int64)
    settings = dict(corruptions=names, severities=severities, seed=7,
                    include_per_class=True)
    expected = evaluate(pure, [image], [mask], 2, **settings)
    first = evaluate(inplace, [image], [mask], 2, **settings)
    second = evaluate(inplace, [image], [mask], 2, **settings)
    assert first == expected == second
    assert len(seen) == 2 * (1 + len(names) * len(severities))
    np.testing.assert_array_equal(seen[0], image)
    np.testing.assert_array_equal(image, np.full_like(image, 180))


def test_preprocessing_does_not_modify_repeated_dataset_aliases():
    image = np.ones((8, 8, 3), dtype=np.uint8)
    mask = image[..., 0]
    observed = []
    def model(im):
        observed.append(int(im[0, 0, 0]))
        im[:] = 0
        return np.zeros(im.shape[:2], dtype=np.int64)
    result = evaluate(model, [image, image], [mask, mask], 2,
                      corruptions=[], severities=[1])
    assert observed == [1, 1]
    assert result['clean_miou'] == 0.
    assert np.all(image == 1)


def test_model_exception_does_not_leave_caller_input_rewritten():
    image = np.full((8, 8, 3), 180, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.int64)
    def model(im):
        im[:] = 0
        raise RuntimeError('synthetic model exception')
    with pytest.raises(RuntimeError, match='synthetic model exception'):
        evaluate(model, [image], [mask], 2, corruptions=[], severities=[1])
    assert np.all(image == 180)


def test_model_can_return_view_of_callback_owned_input():
    image = np.ones((8, 8, 3), dtype=np.uint8)
    target = np.ones((8, 8), dtype=np.int64)
    def model(im):
        im[:] = 0
        return im[..., 0]
    result = evaluate(model, [image], [target], 2, corruptions=[], severities=[1])
    assert result['clean_miou'] == 0.
    assert np.all(image == 1)


def test_aliased_target_is_not_rewritten_into_a_false_perfect_score():
    image = np.ones((8, 8, 3), dtype=np.uint8)
    target = image[..., 0]
    def model(im):
        im[:] = 0
        return np.zeros(im.shape[:2], dtype=np.int64)
    result = evaluate(model, [image], [target], 2, corruptions=[], severities=[1])
    assert result['clean_miou'] == 0.0
    assert np.all(target == 1)
