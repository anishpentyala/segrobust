"""Repeated plan entries cannot silently overwrite observations or weight reports."""
import copy
import json

import numpy as np
import pytest

from segrobust import evaluate, format_report
from segrobust.cli import main


@pytest.mark.parametrize('names,severities', [
    (['brightness'], [1, 1, 5]),
    (['brightness'], [np.int32(1), np.int64(1)]),
    (['brightness', 'contrast', 'brightness'], [1]),
])
@pytest.mark.parametrize('as_generator', [False, True])
def test_duplicate_plan_rejected_before_any_callback(names, severities, as_generator):
    calls = []
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.int64)
    def model(im):
        calls.append('model')
        return mask.copy()
    if as_generator:
        names, severities = iter(names), iter(severities)
    with pytest.raises(ValueError, match='unique'):
        evaluate(model, [image], [mask], 2, corruptions=names,
                 severities=severities, progress=calls.append)
    assert calls == []


def saved_result():
    return {'clean_miou': 1., 'per_corruption': {'brightness': {1: 0., 5: 1.}},
            'mean_corrupt_miou': .5, 'relative_drop': .5, 'n_images': 1,
            'corruptions': ['brightness'], 'severities': [1, 5]}


@pytest.mark.parametrize('field,values', [
    ('severities', [1, 1, 5]),
    ('severities', [np.int32(1), np.int64(1), 5]),
    ('corruptions', ['brightness', 'brightness']),
])
def test_direct_report_rejects_duplicates_without_mutating_result(field, values):
    r = saved_result()
    r[field] = values
    before = copy.deepcopy(r)
    with pytest.raises(ValueError, match='unique'):
        format_report(r)
    assert r == before


@pytest.mark.parametrize('field,values', [
    ('severities', [1, 1, 5]), ('corruptions', ['brightness', 'brightness'])
])
def test_cli_report_rejects_duplicate_plan_without_partial_table(tmp_path, capsys, field, values):
    r = saved_result()
    r[field] = values
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='unique'):
        main(['report', str(path)])
    assert capsys.readouterr().out == ''


def test_unique_unsorted_plan_retains_order_and_exact_model_call_count():
    calls = []
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.int64)
    def model(im):
        calls.append(1)
        return mask.copy()
    r = evaluate(model, [image], [mask], 2,
                 corruptions=iter(['contrast', 'brightness']),
                 severities=iter([np.int32(5), np.int64(1)]))
    assert calls == [1] * 5
    assert r['corruptions'] == ['contrast', 'brightness']
    assert r['severities'] == [5, 1]
    assert r['mean_corrupt_miou'] == 1.
    assert '| corruption | s5 | s1 | mean |' in format_report(r)


def test_empty_corruption_plan_still_supports_clean_only():
    mask = np.zeros((8, 8), dtype=np.int64)
    calls = []
    def model(im):
        calls.append(1)
        return mask.copy()
    r = evaluate(model, [np.zeros((8, 8, 3), dtype=np.uint8)], [mask], 2,
                 corruptions=[], severities=[1])
    assert calls == [1]
    assert r['per_corruption'] == {}
    assert r['clean_miou'] == 1.
    assert np.isnan(r['mean_corrupt_miou'])
