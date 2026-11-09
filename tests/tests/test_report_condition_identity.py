"""A report must not discard saved corruption rows outside its declared plan."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from segrobust import evaluate, format_report
from segrobust.cli import main


def result():
    return {'clean_miou': 1.0, 'mean_corrupt_miou': .5, 'relative_drop': .5,
            'n_images': 1, 'corruptions': ['brightness', 'fog'], 'severities': [1],
            'per_corruption': {'brightness': {1: 1.0}, 'fog': {1: 0.0}}}


@pytest.mark.parametrize('names', [['brightness'], [], ['fog'], ['brightness', 'fog', 'snow']])
def test_direct_list_and_map_must_have_same_conditions(names):
    data = result()
    data['corruptions'] = names
    before = copy.deepcopy(data)
    with pytest.raises(ValueError, match='corruption'):
        format_report(data)
    assert data == before


@pytest.mark.parametrize('names', [['brightness'], [], ['fog'], ['brightness', 'fog', 'snow']])
def test_cli_does_not_silently_drop_or_invent_condition_rows(tmp_path, capsys, names):
    data = result()
    data['corruptions'] = names
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='corruption'):
        main(['report', str(path)])
    assert capsys.readouterr().out == ''
    assert path.read_bytes() == before


@pytest.mark.parametrize('names', [['brightness'], []])
def test_process_exits_nonzero_without_partial_table(tmp_path, names):
    import segrobust
    data = result()
    data['corruptions'] = names
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(data))
    env = dict(os.environ, PYTHONPATH=str(Path(segrobust.__file__).resolve().parent.parent))
    completed = subprocess.run([sys.executable, '-m', 'segrobust.cli', 'report', str(path)],
                               cwd=tmp_path, env=env, text=True, capture_output=True, timeout=10)
    assert completed.returncode != 0
    assert completed.stdout == ''
    assert 'ValueError' in completed.stderr


@pytest.mark.parametrize('names', [['brightness', 'fog'], ['fog', 'brightness'],
                                  ('brightness', 'fog')])
def test_valid_order_controls_are_unchanged(names):
    data = result()
    data['corruptions'] = names
    before = copy.deepcopy(data)
    lines = format_report(data).splitlines()
    assert [line.split('|')[1].strip() for line in lines[2:4]] == list(names)
    assert data == before


@pytest.mark.parametrize('names', [[], ['brightness'], ['contrast', 'brightness']])
@pytest.mark.parametrize('all_ignored', [False, True])
def test_real_evaluate_and_saved_roundtrip_keep_subset_scope(tmp_path, capsys, names, all_ignored):
    image = np.full((8, 8, 3), 120, dtype=np.uint8)
    target = np.full((8, 8), 255 if all_ignored else 0, dtype=np.int64)
    data = evaluate(lambda _: np.zeros((8, 8), dtype=np.int64), [image], [target], 2,
                    corruptions=names, severities=[3, 1], ignore_index=255, seed=11)
    expected = format_report(data)
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(data))
    assert main(['report', str(path)]) == 0
    assert capsys.readouterr().out == expected + '\n'
