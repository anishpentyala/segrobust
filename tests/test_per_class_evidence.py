"""Opt-in class diagnostics use exactly the confusion counts already scored."""
import json

import numpy as np
import pytest

from segrobust import evaluate, format_report
from segrobust import corrupt_pair
from segrobust.metrics import confusion_matrix, iou_from_confusion


def data():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = np.arange(64).reshape(8, 8) * 4
    mask = (image[..., 0] >= 128).astype(np.int64)
    return image, mask


def model(image):
    return (image[..., 0] >= 160).astype(np.int64)


def assert_evidence(actual, cm):
    np.testing.assert_allclose(actual['iou'], iou_from_confusion(cm), equal_nan=True)
    assert actual['target_pixels'] == cm.sum(axis=1).tolist()
    assert all(type(value) is int for value in actual['target_pixels'])


def test_clean_and_real_corruption_match_independent_matrix_oracle():
    import zlib
    image, mask = data()
    r = evaluate(model, [image], [mask], 3, corruptions=['brightness', 'elastic_transform'],
                 severities=[1, 3], seed=7, include_per_class=True)
    assert_evidence(r['per_class']['clean'], confusion_matrix(model(image), mask, 3))
    for name in r['corruptions']:
        for severity in r['severities']:
            draw_seed = (7 * 1_000_003 + zlib.crc32(name.encode()) % 1_000_003 * 7 + severity) % (2 ** 31)
            corrupted, target = corrupt_pair(image, mask, name, severity, seed=draw_seed)
            assert_evidence(r['per_class']['per_corruption'][name][severity],
                            confusion_matrix(model(corrupted), target, 3))


def test_default_exact_result_and_callback_count_unchanged():
    image, mask = data()
    counts = []
    def counted(image):
        counts.append(1)
        return model(image)
    kwargs = dict(corruptions=['brightness'], severities=[1, 3], seed=2)
    default = evaluate(counted, [image], [mask], 2, **kwargs)
    assert len(counts) == 3 and 'per_class' not in default
    counts.clear()
    detailed = evaluate(counted, [image], [mask], 2, include_per_class=True, **kwargs)
    assert len(counts) == 3
    detailed.pop('per_class')
    assert detailed == default


def test_multiple_images_accumulate_pixels_before_iou():
    image, mask = data()
    mask2 = np.ones_like(mask)
    r = evaluate(model, [image, image], [mask, mask2], 3, corruptions=[], include_per_class=True)
    cm = confusion_matrix(model(image), mask, 3) + confusion_matrix(model(image), mask2, 3)
    assert_evidence(r['per_class']['clean'], cm)
    assert r['per_class']['per_corruption'] == {}


def test_ignored_targets_and_absent_class_are_not_zero_scores():
    image, mask = data()
    mask[:] = 255
    r = evaluate(model, [image], [mask], 3, corruptions=['brightness'], severities=[1],
                 ignore_index=255, include_per_class=True)
    for evidence in (r['per_class']['clean'], r['per_class']['per_corruption']['brightness'][1]):
        assert evidence['target_pixels'] == [0, 0, 0]
        assert np.isnan(evidence['iou']).all()


def test_false_positive_only_class_has_zero_iou_but_zero_target_support():
    image, mask = data(); mask[:] = 0
    r = evaluate(lambda im: np.ones(im.shape[:2], dtype=np.int64), [image], [mask], 3,
                 corruptions=[], include_per_class=True)
    assert r['per_class']['clean']['target_pixels'] == [64, 0, 0]
    np.testing.assert_allclose(r['per_class']['clean']['iou'], [0, 0, np.nan], equal_nan=True)
    assert r['clean_miou'] == 0


def test_extended_result_serializes_and_existing_table_stays_same():
    image, mask = data()
    r = evaluate(model, [image], [mask], 2, corruptions=['brightness'], severities=[1], include_per_class=True)
    assert json.loads(json.dumps(r))['per_class']['clean']['target_pixels'] == [32, 32]
    before = format_report(r)
    r.pop('per_class')
    assert format_report(r) == before


@pytest.mark.parametrize('invalid', [None, 1, 0, 'yes', [], np.bool_(True)])
def test_invalid_option_before_any_callbacks(invalid):
    image, mask = data(); calls = []
    with pytest.raises(ValueError, match='include_per_class'):
        evaluate(lambda im: calls.append('model'), [image], [mask], 2,
                 progress=calls.append, include_per_class=invalid)
    assert calls == []
