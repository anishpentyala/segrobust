"""Reports must not relabel experiment settings or discard saved columns."""
import copy
import json

import numpy as np
import pytest

from segrobust import evaluate, format_report
from segrobust.cli import main


def result():
    return {'clean_miou': .9, 'per_corruption': {'fog': {1: .8, 3: .6}},
            'mean_corrupt_miou': .7, 'relative_drop': .2222, 'n_images': 4,
            'corruptions': ['fog'], 'severities': [1, 3]}


@pytest.mark.parametrize('value', [1.9, 1.0, True, False, np.bool_(True),
                                   '1', '1.9', None, 0, 6])
def test_direct_report_rejects_inexact_columns(value):
    r = result()
    r['severities'] = [value]
    r['per_corruption'] = {'fog': {value: .8}}
    with pytest.raises(ValueError, match='severit'):
        format_report(r)


@pytest.mark.parametrize('values', [[]])
def test_direct_report_requires_nonempty_columns(values):
    r = result()
    r['severities'] = values
    with pytest.raises(ValueError, match='nonempty'):
        format_report(r)


@pytest.mark.parametrize('row', [{1: .8}, {1: .8, 3: .6, 5: .2},
                                {True: .8, 3: .6}, {1.0: .8, 3: .6},
                                {'1': .8, '3': .6}])
def test_direct_report_requires_exact_row_keys(row):
    r = result()
    r['per_corruption']['fog'] = row
    with pytest.raises(ValueError, match='severit'):
        format_report(r)


def test_direct_report_accepts_numpy_integer_settings_without_mutation():
    r = result()
    r['severities'] = [np.int64(3), np.int32(1)]
    before = copy.deepcopy(r)
    table = format_report(r)
    assert table.splitlines()[0] == '| corruption | s3 | s1 | mean |'
    assert '| fog | 0.6000 | 0.8000 | 0.7000 |' in table
    assert r == before


def cli_report(tmp_path, data):
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(data))
    return main(['report', str(path)])


@pytest.mark.parametrize('value', [1.9, 1.0, True, '1', 0, 6])
def test_cli_does_not_coerce_columns(tmp_path, capsys, value):
    r = result()
    r['severities'] = [value]
    r['per_corruption'] = {'fog': {'1': .8}}
    with pytest.raises(ValueError, match='severit'):
        cli_report(tmp_path, r)
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('key', ['01', '+1', ' 1', '1 ', '1.0', '0', '6', '-1'])
def test_cli_does_not_normalize_key_aliases(tmp_path, capsys, key):
    r = result()
    r['per_corruption']['fog'][key] = .2
    with pytest.raises(ValueError, match='canonical'):
        cli_report(tmp_path, r)
    assert capsys.readouterr().out == ''


def test_cli_rejects_duplicate_raw_json_keys(tmp_path, capsys):
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(result()).replace('"1": 0.8', '"1": 0.8, "1": 0.2'))
    with pytest.raises(ValueError, match='duplicate JSON key'):
        main(['report', str(path)])
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('severities', [[1, 3], [5, 2, 1], [3, 1, 3]])
def test_real_evaluate_json_roundtrip(tmp_path, capsys, severities):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.int64)
    r = evaluate(lambda im: np.zeros(im.shape[:2], dtype=np.int64),
                 [img], [mask], 2, corruptions=['brightness', 'contrast'],
                 severities=severities, seed=7)
    expected = format_report(r)
    before = copy.deepcopy(r)
    assert cli_report(tmp_path, r) == 0
    assert capsys.readouterr().out == expected + '\n'
    assert r == before


@pytest.mark.parametrize('kind', ['fractional', 'alias', 'duplicate'])
def test_cli_process_fails_without_partial_table(tmp_path, kind):
    import os
    import subprocess
    import sys
    from pathlib import Path

    r = result()
    if kind == 'fractional':
        r['severities'] = [1.9, 3]
    elif kind == 'alias':
        r['per_corruption']['fog']['01'] = .2
    path = tmp_path / 'result.json'
    data = json.dumps(r)
    if kind == 'duplicate':
        data = data.replace('"1": 0.8', '"1": 0.8, "1": 0.2')
    path.write_text(data)
    env = dict(os.environ)
    # Select the same source as this test process, including installed runs.
    import segrobust
    env['PYTHONPATH'] = str(Path(segrobust.__file__).resolve().parent.parent)
    completed = subprocess.run([sys.executable, '-m', 'segrobust.cli', 'report',
                                str(path)], cwd=tmp_path, env=env,
                               text=True, capture_output=True, timeout=10)
    assert completed.returncode != 0
    assert completed.stdout == ''
    assert 'ValueError' in completed.stderr


def test_direct_report_integer_index_columns_and_rows():
    class Setting:
        def __init__(self, value):
            self.value = value

        def __index__(self):
            return self.value

    r = result()
    r['severities'] = [Setting(3), np.int32(1)]
    r['per_corruption']['fog'] = {np.int32(1): .8, np.int64(3): .6}
    assert '| fog | 0.6000 | 0.8000 | 0.7000 |' in format_report(r)


def test_rejection_does_not_rewrite_result():
    r = result()
    r['severities'] = [1.9, 3]
    before = copy.deepcopy(r)
    with pytest.raises(ValueError):
        format_report(r)
    assert r == before


@pytest.mark.parametrize('row', [{'1': .8}, {'1': .8, '3': .6, '5': .2}])
def test_cli_rejects_missing_or_extra_settings_without_table(tmp_path, capsys, row):
    r = result()
    r['per_corruption']['fog'] = row
    with pytest.raises(ValueError, match='match report columns'):
        cli_report(tmp_path, r)
    assert capsys.readouterr().out == ''


def test_cli_duplicate_top_level_setting_list_is_rejected(tmp_path, capsys):
    path = tmp_path / 'result.json'
    data = json.dumps(result()).replace('"severities": [1, 3]',
                                       '"severities": [1, 3], "severities": [1]')
    path.write_text(data)
    with pytest.raises(ValueError, match='duplicate JSON key'):
        main(['report', str(path)])
    assert capsys.readouterr().out == ''


def test_all_short_valid_setting_orders_keep_exact_columns():
    from itertools import product

    for length in (1, 2, 3):
        for settings in product(range(1, 6), repeat=length):
            r = result()
            r['severities'] = settings
            r['per_corruption']['fog'] = {s: s / 10 for s in settings}
            table = format_report(r)
            expected_header = '| corruption | ' + ' | '.join(
                f's{s}' for s in settings) + ' | mean |'
            expected_cells = ' | '.join(f'{s / 10:.4f}' for s in settings)
            assert table.splitlines()[0] == expected_header
            assert table.splitlines()[2].startswith('| fog | ' + expected_cells + ' | ')


@pytest.mark.parametrize('settings', [[5, 1, 5], [2, 4, 3]])
def test_json_report_keeps_nan_from_all_ignored_targets(tmp_path, capsys, settings):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.full((8, 8), 255, dtype=np.int64)
    r = evaluate(lambda im: np.zeros(im.shape[:2], dtype=np.int64),
                 [img], [mask], 2, corruptions=['brightness'],
                 severities=settings, seed=7, ignore_index=255)
    assert np.isnan(r['clean_miou'])
    assert np.isnan(r['mean_corrupt_miou'])
    expected = format_report(r)
    path = tmp_path / 'ignored.json'
    path.write_text(json.dumps(r))
    before = path.read_bytes()
    assert main(['report', str(path)]) == 0
    assert capsys.readouterr().out == expected + '\n'
    assert path.read_bytes() == before


def test_empty_corruption_selection_keeps_existing_nan_report(tmp_path, capsys):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.int64)
    r = evaluate(lambda im: np.zeros(im.shape[:2], dtype=np.int64),
                 [img], [mask], 2, corruptions=[], severities=[3, 1])
    assert r['per_corruption'] == {}
    expected = format_report(r)
    assert cli_report(tmp_path, r) == 0
    assert capsys.readouterr().out == expected + '\n'


def test_escaped_duplicate_json_keys_are_not_a_bypass(tmp_path, capsys):
    path = tmp_path / 'escaped.json'
    data = json.dumps(result()).replace('"1": 0.8', '"1": 0.8, "\\u0031": 0.2')
    path.write_text(data)
    with pytest.raises(ValueError, match='duplicate JSON key'):
        main(['report', str(path)])
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('key', ['１', '١', '1e0', '\t1', '1\n'])
def test_unicode_and_numeric_aliases_do_not_replace_saved_settings(tmp_path, capsys, key):
    r = result()
    r['per_corruption']['fog'][key] = .2
    with pytest.raises(ValueError, match='canonical'):
        cli_report(tmp_path, r)
    assert capsys.readouterr().out == ''
