"""Segmentation metrics: confusion-matrix accumulation and mIoU."""

import numpy as np


def confusion_matrix(pred, target, num_classes, ignore_index=None):
    """Accumulate a (num_classes, num_classes) confusion matrix.

    Rows are ground-truth labels, columns are predictions. Pixels whose
    target is ignore_index or out of range ([0, num_classes)) are
    excluded. A prediction label out of range at a pixel whose target IS
    valid raises ValueError: clipping it into range could manufacture
    correct predictions, and dropping it would shrink the denominator and
    can inflate the score, so neither is done silently.
    """
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    if pred.shape != target.shape:
        raise ValueError('pred and target must have the same number of pixels')
    valid = (target >= 0) & (target < num_classes)
    if ignore_index is not None:
        valid &= target != ignore_index
    # Out-of-range predictions at valid-target pixels are an ERROR, never
    # silently clipped into a valid class (which can manufacture correct
    # predictions) and never dropped (which shrinks the denominator and can
    # inflate the score). The caller must fix or deliberately map the
    # model's output before scoring.
    bad = valid & ((pred < 0) | (pred >= num_classes))
    if bad.any():
        examples = sorted({int(v) for v in pred[bad]})[:5]
        raise ValueError(
            f'prediction labels out of range [0, {num_classes}) at '
            f'{int(bad.sum())} pixels (values include {examples}); fix the '
            'model output or map it into range deliberately before scoring'
        )
    label = num_classes * target[valid] + pred[valid]
    return np.bincount(label, minlength=num_classes ** 2).reshape(
        num_classes, num_classes)


def iou_from_confusion(cm):
    """Per-class IoU from a confusion matrix; NaN for absent classes."""
    tp = np.diag(cm).astype(np.float64)
    union = (cm.sum(axis=1) + cm.sum(axis=0) - tp).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(union > 0, tp / union, np.nan)


def miou_from_confusion(cm):
    """Mean IoU over classes present in the ground truth."""
    ious = iou_from_confusion(cm)
    present = cm.sum(axis=1) > 0
    if not present.any():
        return float('nan')
    return float(np.nanmean(ious[present]))
