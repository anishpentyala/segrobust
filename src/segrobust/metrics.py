"""Segmentation metrics: confusion-matrix accumulation and mIoU."""

import numpy as np


def confusion_matrix(pred, target, num_classes, ignore_index=None):
    """Accumulate a (num_classes, num_classes) confusion matrix.

    Rows are ground-truth labels, columns are predictions. Pixels labeled
    ignore_index in the target are excluded, as are out-of-range labels.
    """
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    if pred.shape != target.shape:
        raise ValueError('pred and target must have the same number of pixels')
    valid = (target >= 0) & (target < num_classes)
    if ignore_index is not None:
        valid &= target != ignore_index
    label = num_classes * target[valid] + pred[valid].clip(0, num_classes - 1)
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
