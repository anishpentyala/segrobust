import numpy as np

from segrobust import evaluate, format_report


def synthetic_dataset(n=3, h=32, w=32, classes=2, seed=0):
    rng = np.random.RandomState(seed)
    images, masks = [], []
    for _ in range(n):
        # clean signal: label depends only on pixel intensity ordering
        img = rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
        mask = (img[..., 0] > img[..., 1]).astype(np.int64)
        images.append(img)
        masks.append(mask)
    return images, masks


def good_model(img):
    return (img[..., 0] > img[..., 1]).astype(np.int64)


def constant_model(img):
    return np.zeros(img.shape[:2], dtype=np.int64)


def test_evaluate_clean_perfect_model():
    images, masks = synthetic_dataset()
    res = evaluate(good_model, images, masks, num_classes=2,
                   corruptions=['brightness'], severities=[1, 2])
    assert res['clean_miou'] == 1.0
    # brightness is monotone per-channel-ish; a perfect model should stay high
    assert res['per_corruption']['brightness'][1] > 0.9


def test_evaluate_bad_model_drops():
    images, masks = synthetic_dataset()
    res = evaluate(constant_model, images, masks, num_classes=2,
                   corruptions=['gaussian_noise'], severities=[3])
    assert res['clean_miou'] < 1.0
    assert 0.0 <= res['mean_corrupt_miou'] <= 1.0


def test_evaluate_is_reproducible():
    images, masks = synthetic_dataset()
    r1 = evaluate(good_model, images, masks, 2,
                  corruptions=['gaussian_noise', 'fog'], severities=[2, 4])
    r2 = evaluate(good_model, images, masks, 2,
                  corruptions=['gaussian_noise', 'fog'], severities=[2, 4])
    assert r1 == r2


def test_evaluate_structure_and_report():
    images, masks = synthetic_dataset()
    res = evaluate(good_model, images, masks, 2,
                   corruptions='validation', severities=[1])
    assert set(res['per_corruption']) == set(res['corruptions'])
    table = format_report(res)
    assert '| corruption | s1 | mean |' in table
    assert 'clean mIoU' in table


def test_evaluate_shape_mismatch_raises():
    images, masks = synthetic_dataset()
    bad = lambda img: np.zeros((4, 4), dtype=np.int64)
    try:
        evaluate(bad, images, masks, 2, corruptions=['fog'], severities=[1])
    except ValueError as e:
        assert 'shape' in str(e)
    else:
        raise AssertionError('expected ValueError')
