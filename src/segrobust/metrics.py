"""Segmentation metrics: confusion-matrix accumulation and mIoU."""

import math
import operator

import numpy as np


def _class_count(num_classes):
    """Validate class count before allocation or user model callbacks."""
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
    return num_classes


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
    num_classes = _class_count(num_classes)
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


def _confusion_values(cm):
    """A square, finite, nonnegative count/weight table, widened before sums."""
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError('confusion matrix must be square and two-dimensional')
    if cm.dtype.kind not in 'biuf':
        raise TypeError('confusion matrix must contain real numeric counts')
    values = cm.astype(np.result_type(cm.dtype, np.float64))
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('confusion matrix counts must be finite and nonnegative')
    return values


def iou_from_confusion(cm):
    """Per-class IoU from finite nonnegative counts/weights; NaN if absent."""
    values = _confusion_values(cm)
    tp = np.diag(values)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        union = values.sum(axis=1) + values.sum(axis=0) - tp
        result = np.where(union > 0, tp / union, np.nan)
        overflowing = ~np.isfinite(union)
        if overflowing.any():
            # Scale each affected class independently: an unrelated large
            # class must not erase a small class's counts by underflow.
            scale = np.maximum(values.max(axis=1), values.max(axis=0))
            rows = values[overflowing] / scale[overflowing, None]
            columns = values[:, overflowing] / scale[None, overflowing]
            diagonal = tp[overflowing] / scale[overflowing]
            scaled_union = rows.sum(axis=1) + columns.sum(axis=0) - diagonal
            result[overflowing] = diagonal / scaled_union
    return result



def miou_from_confusion(cm):
    """Mean IoU over classes present in the ground truth."""
    values = _confusion_values(cm)
    ious = iou_from_confusion(values)
    present = (values > 0).any(axis=1)
    if not present.any():
        return float('nan')
    return float(np.nanmean(ious[present]))
