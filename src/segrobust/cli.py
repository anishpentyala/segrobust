"""segrobust command line interface."""

import argparse
import json
import sys

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='segrobust',
        description='Natural-corruption robustness evaluation for '
                    'segmentation models.')
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('list', help='list available corruptions')

    p_corrupt = sub.add_parser('corrupt', help='corrupt one image file')
    p_corrupt.add_argument('image', help='input image (png/jpg)')
    p_corrupt.add_argument('--name', required=True, help='corruption name')
    p_corrupt.add_argument('--severity', type=int, default=1)
    p_corrupt.add_argument('--seed', type=int, default=None)
    p_corrupt.add_argument('--out', required=True, help='output image path')

    p_report = sub.add_parser('report',
                              help='render an evaluate() JSON result as '
                                   'a markdown table')
    p_report.add_argument('json_path', help='JSON file from evaluate()')

    args = parser.parse_args(argv)

    if args.command == 'list':
        from .api import CORRUPTIONS
        for name, meta in CORRUPTIONS.items():
            tags = [meta['category']]
            if meta['geometric']:
                tags.append('geometric')
            if meta['validation']:
                tags.append('validation-only')
            print(f'{name:20s} [{", ".join(tags)}]')
        return 0

    if args.command == 'corrupt':
        from PIL import Image
        from .api import corrupt
        img = np.array(Image.open(args.image).convert('RGB'))
        out = corrupt(img, args.name, severity=args.severity, seed=args.seed)
        Image.fromarray(out).save(args.out)
        print(f'wrote {args.out} ({args.name} severity {args.severity})')
        return 0

    if args.command == 'report':
        from .evaluate import format_report
        with open(args.json_path) as fh:
            result = json.load(fh)
        # JSON keys arrive as strings; restore int severity keys
        result['severities'] = [int(s) for s in result['severities']]
        result['per_corruption'] = {
            k: {int(s): v for s, v in row.items()}
            for k, row in result['per_corruption'].items()}
        print(format_report(result))
        return 0

    return 2


if __name__ == '__main__':
    sys.exit(main())
