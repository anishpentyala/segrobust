"""Explicit same-target controls for silent report setting coercion."""
import json

from segrobust import format_report
from segrobust.cli import main


def report():
    return {'clean_miou': .9, 'mean_corrupt_miou': .7, 'relative_drop': .2,
            'n_images': 1, 'corruptions': ['fog'], 'severities': [1.9],
            'per_corruption': {'fog': {1.9: .7}}}


def test_fractional_direct_setting_is_not_presented_as_s1():
    try:
        format_report(report())
    except ValueError:
        return
    raise AssertionError('fractional severity must not be presented as s1')


def test_fractional_saved_setting_is_not_presented_as_s1(tmp_path, capsys):
    r = report()
    r['per_corruption'] = {'fog': {'1': .7}}
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(r))
    try:
        main(['report', str(path)])
    except ValueError:
        assert capsys.readouterr().out == ''
        return
    raise AssertionError('saved fractional severity must not be presented as s1')


def test_saved_alias_columns_are_not_silently_discarded(tmp_path, capsys):
    r = report()
    r['severities'] = [1]
    r['per_corruption'] = {'fog': {'01': .7, '1': .2}}
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(r))
    try:
        main(['report', str(path)])
    except ValueError:
        assert capsys.readouterr().out == ''
        return
    raise AssertionError('saved alias columns must not silently overwrite scores')
