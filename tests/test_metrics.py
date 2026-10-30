import numpy as np

from segrobust import confusion_matrix, iou_from_confusion, miou_from_confusion


def test_confusion_matrix_counts():
    pred = np.array([[0, 1], [1, 0]])
    tgt = np.array([[0, 1], [0, 1]])
    cm = confusion_matrix(pred, tgt, 2)
    assert cm.tolist() == [[1, 1], [1, 1]]


def test_miou_perfect_and_half():
    tgt = np.array([[0, 1], [1, 0]])
    assert miou_from_confusion(confusion_matrix(tgt, tgt, 2)) == 1.0
    pred = np.array([[0, 0], [0, 0]])
    # class 0: tp=2 fp=2 fn=0 -> 2/4 = 0.5 ; class 1: tp=0 union=2 -> 0
    assert miou_from_confusion(confusion_matrix(pred, tgt, 2)) == (0.5 + 0) / 2


def test_ignore_index_excluded():
    pred = np.array([0, 1, 1])
    tgt = np.array([0, 255, 0])
    cm = confusion_matrix(pred, tgt, 2, ignore_index=255)
    assert cm.sum() == 2


def test_absent_class_nan_not_zero():
    pred = tgt = np.zeros((4, 4), dtype=np.int64)
    cm = confusion_matrix(pred, tgt, 3)
    ious = iou_from_confusion(cm)
    assert ious[0] == 1.0 and np.isnan(ious[1]) and np.isnan(ious[2])
    assert miou_from_confusion(cm) == 1.0


def test_out_of_range_predictions_raise_not_clipped():
    """Regression (audit S2 + review): pred=5 with num_classes=3 must not
    be clipped into class 2 (manufactures correctness) nor dropped
    (shrinks the denominator, can inflate the score) - it raises."""
    import pytest
    with pytest.raises(ValueError, match="out of range"):
        confusion_matrix(np.array([0, 1, 2, 5]), np.array([0, 1, 2, 2]), 3)
    with pytest.raises(ValueError, match="out of range"):
        confusion_matrix(np.array([0, -1]), np.array([0, 0]), 3)
    # invalid preds at IGNORED/out-of-range targets are excluded, not errors
    cm = confusion_matrix(np.array([0, 5]), np.array([0, 255]), 3,
                          ignore_index=255)
    assert cm.sum() == 1
    cm2 = confusion_matrix(np.array([0, 5]), np.array([0, 7]), 3)
    assert cm2.sum() == 1
