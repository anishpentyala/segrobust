"""Corruption-robustness evaluation for segmentation models.

The model is a plain callable: image (uint8 HxWx3) -> predicted label map
(int HxW). Nothing framework-specific is imported here.
"""

import zlib

import numpy as np

from .api import corrupt_pair, get_corruption_names
from .metrics import confusion_matrix, miou_from_confusion


def evaluate(model_fn, images, masks, num_classes, corruptions='common',
             severities=(1, 2, 3, 4, 5), seed=0, ignore_index=None,
             progress=None):
    """Evaluate mIoU on clean images and under each corruption/severity.

    model_fn: callable uint8 image -> int label map (H, W).
    images, masks: equal-length sequences of uint8 images and int masks.
    corruptions: subset name ('common', 'validation', 'all', a category)
        or an explicit list of corruption names.
    severities: iterable of ints in [1, 5].
    seed: base seed; the (image index, corruption, severity) triple maps
        deterministically to one corruption draw, so reruns reproduce.
    progress: optional callable(msg: str).

    Returns a dict:
      clean_miou: float
      per_corruption: {name: {severity: miou}}
      mean_corrupt_miou: mean over corruptions and severities
      relative_drop: (clean_miou - mean_corrupt_miou) / clean_miou
    """
    images = list(images)
    masks = list(masks)
    if len(images) != len(masks):
        raise ValueError('images and masks must have the same length')
    if not images:
        raise ValueError('need at least one image')
    if isinstance(corruptions, str):
        names = get_corruption_names(corruptions)
    else:
        names = list(corruptions)
    severities = [int(s) for s in severities]

    def run_batch(transform=None):
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for i, (img, msk) in enumerate(zip(images, masks)):
            im = img
            mk = msk
            if transform is not None:
                im, mk = transform(i, img, msk)
            pred = np.asarray(model_fn(im))
            if pred.shape[:2] != mk.shape[:2]:
                raise ValueError(
                    f'model_fn returned shape {pred.shape} for image '
                    f'{i}, expected spatial shape {mk.shape[:2]}')
            cm += confusion_matrix(pred, mk, num_classes,
                                   ignore_index=ignore_index)
        return miou_from_confusion(cm)

    if progress:
        progress('clean pass')
    clean = run_batch()

    per_corruption = {}
    for name in names:
        per_corruption[name] = {}
        for sev in severities:
            if progress:
                progress(f'{name} severity {sev}')

            def transform(i, img, msk, name=name, sev=sev):
                # one deterministic draw per (image, corruption, severity)
                draw_seed = (seed * 1_000_003 + i * 1_009 +
                             zlib.crc32(name.encode()) % 1_000_003 * 7 + sev) % (2 ** 31)
                return corrupt_pair(img, msk, name, sev, seed=draw_seed,
                                    ignore_index=ignore_index)

            per_corruption[name][sev] = run_batch(transform)

    flat = [v for per_sev in per_corruption.values() for v in per_sev.values()]
    mean_corrupt = float(np.mean(flat)) if flat else float('nan')
    rel_drop = ((clean - mean_corrupt) / clean
                if clean and not np.isnan(clean) and clean != 0 else float('nan'))
    return {
        'clean_miou': clean,
        'per_corruption': per_corruption,
        'mean_corrupt_miou': mean_corrupt,
        'relative_drop': rel_drop,
        'n_images': len(images),
        'corruptions': names,
        'severities': severities,
    }


def format_report(result):
    """Render an evaluate() result as a markdown table."""
    lines = ['| corruption | ' + ' | '.join(f's{int(s)}'
             for s in result['severities']) + ' | mean |',
             '|' + '---|' * (len(result['severities']) + 2)]
    for name in result['corruptions']:
        row = result['per_corruption'][name]
        vals = [row[s] for s in result['severities']]
        cells = ' | '.join(f'{v:.4f}' for v in vals)
        lines.append(f'| {name} | {cells} | {np.mean(vals):.4f} |')
    lines.append('')
    lines.append(f"clean mIoU: {result['clean_miou']:.4f}  |  "
                 f"mean corrupt mIoU: {result['mean_corrupt_miou']:.4f}  |  "
                 f"relative drop: {result['relative_drop']:.4f}  |  "
                 f"images: {result['n_images']}")
    return '\n'.join(lines)
