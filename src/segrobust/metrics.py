"""Segmentation metrics: confusion-matrix accumulation and mIoU."""

import math
import operator

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
    if pred.dtype.kind not in 'biu' or target.dtype.kind not in 'biu':
        raise TypeError('pred and target labels must have integer dtypes')
    try:
        if isinstance(num_classes, (bool, np.bool_)):
            raise TypeError
        num_classes = operator.index(num_classes)
    except TypeError:
        raise ValueError('num_classes must be a positive integer') from None
    if num_classes <= 0:
        raise ValueError('num_classes must be a positive integer')
    if num_classes > math.isqrt(np.iinfo(np.intp).max):
        raise ValueError('num_classes squared exceeds platform index range')
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
    # Label storage may be uint8/int8 even when num_classes**2 exceeds
    # that dtype. Widen each valid operand before multiplication/addition;
    # mixed uint64/int64 arithmetic must not promote bin indices to float.
    # Do not cast before filtering: ignored uint64 sentinel values can exceed
    # the signed index range. Casting floats would silently truncate labels.
    target_indices = target[valid].astype(np.intp)
    pred_indices = pred[valid].astype(np.intp)
    label = num_classes * target_indices + pred_indices
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
