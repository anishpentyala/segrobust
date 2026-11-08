"""Corruption-robustness evaluation for segmentation models.

The model is a plain callable: image (uint8 HxWx3) -> predicted label map
(int HxW). Nothing framework-specific is imported here.
"""

import operator
import zlib

import numpy as np

from .api import (corrupt_pair, get_corruption_names, _severity_index,
                  _validate_image, _validate_mask, CORRUPTIONS)
from .metrics import confusion_matrix, miou_from_confusion, iou_from_confusion, _class_count


def evaluate(model_fn, images, masks, num_classes, corruptions='common',
             severities=(1, 2, 3, 4, 5), seed=0, ignore_index=None,
             progress=None, include_per_class=False):
    """Evaluate mIoU on clean images and under each corruption/severity.

    model_fn: callable uint8 image -> int label map (H, W). Each call receives
        its own writable image copy, so in-place preprocessing cannot rewrite
        caller images, aliased targets or later corruption inputs. This does
        not isolate model state, captured variables or unrelated concurrent code.
    images, masks: equal-length sequences of uint8 images and int masks.
    corruptions: subset name ('common', 'validation', 'all', a category)
        or an explicit list of corruption names.
    severities: iterable of ints in [1, 5].
    seed: base seed; the (image index, corruption, severity) triple maps
        deterministically to one corruption draw, so reruns reproduce.
    progress: optional callable(msg: str).
    include_per_class: bool, opt into class-indexed IoU and target pixel counts
        from the same pooled confusion matrices, without extra model calls.

    Returns a dict:
      clean_miou: float
      per_corruption: {name: {severity: miou}}
      mean_corrupt_miou: mean over corruptions and severities
      relative_drop: (clean_miou - mean_corrupt_miou) / clean_miou
      per_class (only when requested): clean and per_corruption evidence, each
        with class-indexed iou and target_pixels lists. IoU is NaN only when
        the class has neither target nor prediction pixels. Target-absent
        false-positive classes have zero IoU but remain excluded from mIoU.
    """
    if type(include_per_class) is not bool:
        raise ValueError("include_per_class must be a boolean")
    num_classes = _class_count(num_classes)
    images = list(images)
    masks = list(masks)
    if len(images) != len(masks):
        raise ValueError('images and masks must have the same length')
    if not images:
        raise ValueError('need at least one image')
    # Validate the complete dataset before calling the model or reporting progress.
    for img, msk in zip(images, masks):
        _validate_image(img)
        _validate_mask(msk, img.shape[:2])
    if isinstance(corruptions, str):
        names = get_corruption_names(corruptions)
    else:
        names = list(corruptions)
    for name in names:
        if not isinstance(name, str) or name not in CORRUPTIONS:
            raise ValueError(f"unknown corruption {name!r}")
    if len(set(names)) != len(names):
        raise ValueError("corruption names must be unique")
    severities = [_severity_index(s) for s in severities]
    if not severities:
        raise ValueError("need at least one severity")

    if len(set(severities)) != len(severities):
        raise ValueError("severities must be unique")

    def run_batch(transform=None):
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for i, (img, msk) in enumerate(zip(images, masks)):
            im = img
            mk = msk
            if transform is not None:
                im, mk = transform(i, img, msk)
            # The clean image may be caller-owned or share storage with its
            # target/other samples. Treat model preprocessing as local to this
            # invocation, just as corrupted conditions use independent images.
            pred = np.asarray(model_fn(im.copy()))
            if pred.shape != mk.shape:
                raise ValueError(
                    f'model_fn returned shape {pred.shape} for image '
                    f'{i}, expected spatial shape {mk.shape[:2]}')
            cm += confusion_matrix(pred, mk, num_classes,
                                   ignore_index=ignore_index)
        score = miou_from_confusion(cm)
        evidence = None
        if include_per_class:
            evidence = {"iou": iou_from_confusion(cm).tolist(),
                        "target_pixels": cm.sum(axis=1).tolist()}
        return score, evidence

    if progress:
        progress('clean pass')
    clean, clean_classes = run_batch()
    class_scores = {}

    per_corruption = {}
    for name in names:
        per_corruption[name] = {}
        if include_per_class:
            class_scores[name] = {}
        for sev in severities:
            if progress:
                progress(f'{name} severity {sev}')

            def transform(i, img, msk, name=name, sev=sev):
                # one deterministic draw per (image, corruption, severity)
                draw_seed = (seed * 1_000_003 + i * 1_009 +
                             zlib.crc32(name.encode()) % 1_000_003 * 7 + sev) % (2 ** 31)
                return corrupt_pair(img, msk, name, sev, seed=draw_seed,
                                    ignore_index=ignore_index)

            score, classes = run_batch(transform)
            per_corruption[name][sev] = score
            if include_per_class:
                class_scores[name][sev] = classes

    flat = [v for per_sev in per_corruption.values() for v in per_sev.values()]
    mean_corrupt = float(np.mean(flat)) if flat else float('nan')
    rel_drop = ((clean - mean_corrupt) / clean
                if clean and not np.isnan(clean) and clean != 0 else float('nan'))
    result = {
        'clean_miou': clean,
        'per_corruption': per_corruption,
        'mean_corrupt_miou': mean_corrupt,
        'relative_drop': rel_drop,
        'n_images': len(images),
        'corruptions': names,
        'severities': severities,
    }
    if include_per_class:
        result['per_class'] = {'clean': clean_classes,
                               'per_corruption': class_scores}
    return result


def format_report(result):
    """Render an evaluate() result as a markdown table."""
    # A rendered column is evidence of an exact experiment setting. Never
    # truncate floats or treat True as severity 1 when naming that column.
    severities = []
    for value in result['severities']:
        try:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError
            severity = operator.index(value)
        except TypeError:
            raise ValueError('report severities must be integers from 1 to 5') from None
        if not 1 <= severity <= 5:
            raise ValueError('report severities must be integers from 1 to 5')
        severities.append(severity)
    if not severities:
        raise ValueError('report severities must be nonempty')
    if len(set(severities)) != len(severities):
        raise ValueError('report severities must be unique')
    names = result['corruptions']
    if len(set(names)) != len(names):
        raise ValueError('report corruption names must be unique')
    for name in names:
        row = result['per_corruption'][name]
        # Validate key types as well as equality: True and 1.0 compare equal
        # to 1 in a dict, but do not identify integer experiment settings.
        for key in row:
            if isinstance(key, (bool, np.bool_)):
                raise ValueError('report row severity keys must be integers')
            try:
                operator.index(key)
            except TypeError:
                raise ValueError('report row severity keys must be integers') from None
        if set(row) != set(severities):
            raise ValueError('report row severities must match report columns')
    lines = ['| corruption | ' + ' | '.join(f's{s}'
             for s in severities) + ' | mean |',
             '|' + '---|' * (len(severities) + 2)]
    for name in result['corruptions']:
        row = result['per_corruption'][name]
        vals = [row[s] for s in severities]
        cells = ' | '.join(f'{v:.4f}' for v in vals)
        lines.append(f'| {name} | {cells} | {np.mean(vals):.4f} |')
    lines.append('')
    lines.append(f"clean mIoU: {result['clean_miou']:.4f}  |  "
                 f"mean corrupt mIoU: {result['mean_corrupt_miou']:.4f}  |  "
                 f"relative drop: {result['relative_drop']:.4f}  |  "
                 f"images: {result['n_images']}")
    return '\n'.join(lines)
