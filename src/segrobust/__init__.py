"""segrobust: natural-corruption robustness evaluation for segmentation."""

from .api import (CORRUPTIONS, CORRUPTION_NAMES, GEOMETRIC, corrupt,
                  corrupt_pair, get_corruption_names)
from .evaluate import evaluate, format_report
from .metrics import confusion_matrix, iou_from_confusion, miou_from_confusion

__version__ = '0.1.0'

__all__ = ['CORRUPTIONS', 'CORRUPTION_NAMES', 'GEOMETRIC', 'corrupt',
           'corrupt_pair', 'get_corruption_names', 'evaluate',
           'format_report', 'confusion_matrix', 'iou_from_confusion',
           'miou_from_confusion', '__version__']
