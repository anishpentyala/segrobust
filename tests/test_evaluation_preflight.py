"""Invalid experiment settings must not start expensive model inference."""
import math
import numpy as np
import pytest
from segrobust import evaluate
from segrobust.metrics import confusion_matrix

@pytest.mark.parametrize('count', [0, -1, True, np.bool_(False), 1.5, '2', None,
                                  math.isqrt(np.iinfo(np.intp).max) + 1])
def test_invalid_class_count_before_model_or_progress(count):
    calls = []
    def model(image):
        calls.append('model')
        return np.zeros(image.shape[:2], dtype=np.int64)
    with pytest.raises(ValueError, match='num_classes'):
        evaluate(model, [np.zeros((8, 8, 3), dtype=np.uint8)],
                 [np.zeros((8, 8), dtype=np.int64)], count,
                 corruptions=[], progress=calls.append)
    assert calls == []

@pytest.mark.parametrize('names', [['fog', 'not_a_corruption'], [None], [True],
                                 [1], [[]], [{}]])
def test_invalid_corruption_before_model_or_progress(names):
    calls = []
    def model(image):
        calls.append('model')
        return np.zeros(image.shape[:2], dtype=np.int64)
    with pytest.raises(ValueError, match='unknown corruption'):
        evaluate(model, [np.zeros((8, 8, 3), dtype=np.uint8)],
                 [np.zeros((8, 8), dtype=np.int64)], 2,
                 corruptions=names, severities=[1], progress=calls.append)
    assert calls == []

@pytest.mark.parametrize('count', [1, np.int8(1), np.uint64(1)])
def test_valid_integer_class_count_keeps_clean_result(count):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.int64)
    result = evaluate(lambda _: mask.copy(), [image], [mask], count,
                      corruptions=['brightness'], severities=[1])
    assert result['clean_miou'] == 1.0
    assert result['per_corruption']['brightness'][1] == 1.0
    np.testing.assert_array_equal(confusion_matrix(mask, mask, count), [[64]])

def test_zero_class_count_rejects_before_expensive_model_control():
    """Literal model invocation witness, not an incidental NumPy allocation error."""
    calls = []
    def costly_model(image):
        calls.append('inference')
        return np.zeros(image.shape[:2], dtype=np.int64)
    try:
        evaluate(costly_model, [np.zeros((8, 8, 3), dtype=np.uint8)],
                 [np.zeros((8, 8), dtype=np.int64)], 0, corruptions=[])
    except ValueError:
        pass
    assert calls == [], 'invalid class count must not invoke expensive model inference'


def test_late_unknown_corruption_rejects_before_any_model_control():
    calls = []
    def costly_model(image):
        calls.append('inference')
        return np.zeros(image.shape[:2], dtype=np.int64)
    try:
        evaluate(costly_model, [np.zeros((8, 8, 3), dtype=np.uint8)],
                 [np.zeros((8, 8), dtype=np.int64)], 2,
                 corruptions=['brightness', 'missing'], severities=[1])
    except ValueError:
        pass
    assert calls == [], 'invalid corruption plan must not invoke expensive model inference'
