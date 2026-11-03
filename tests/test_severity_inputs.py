"""Requested severity is never silently changed by int/equality coercion."""
import numpy as np
import pytest
from segrobust import corrupt, corrupt_pair, evaluate

INVALID = [True, np.bool_(False), 1.0, 1.9, np.float64(2), '2', None,
           float('nan'), float('inf'), 0, 6, [1]]


@pytest.mark.parametrize('severity',INVALID)
@pytest.mark.parametrize('name',['brightness','elastic_transform'])
def test_single_and_pair_reject_invalid_severity(severity,name):
    image=np.zeros((8,8,3),dtype=np.uint8)
    mask=np.zeros((8,8),dtype=np.int64)
    with pytest.raises(ValueError,match='severity must be an integer'):
        corrupt(image,name,severity=severity,seed=0)
    with pytest.raises(ValueError,match='severity must be an integer'):
        corrupt_pair(image,mask,name,severity=severity,seed=0)


@pytest.mark.parametrize('severity',INVALID)
def test_evaluation_rejects_before_model_or_progress(severity):
    image=np.zeros((8,8,3),dtype=np.uint8)
    mask=np.zeros((8,8),dtype=np.int64)
    calls=[]
    def model(image):calls.append('model');return mask
    with pytest.raises(ValueError,match='severity must be an integer'):
        evaluate(model,[image],[mask],1,corruptions=['brightness'],
                 severities=[1,severity],progress=calls.append)
    assert calls==[]


class IndexSeverity:
    def __index__(self):return 2


@pytest.mark.parametrize('severity',[1,np.int32(2),np.int64(5),IndexSeverity()])
def test_exact_integer_index_values_keep_requested_severity(severity):
    image=np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3)
    mask=np.zeros((8,8),dtype=np.int64)
    result=evaluate(lambda x:mask,[image],[mask],1,corruptions=['brightness'],severities=[severity])
    expected=severity.__index__() if hasattr(severity,'__index__') else severity
    assert result['severities']==[expected]
    assert type(result['severities'][0]) is int
    assert np.array_equal(corrupt(image,'brightness',severity,seed=4),
                          corrupt(image,'brightness',expected,seed=4))


def test_empty_severity_request_fails_before_model():
    image=np.zeros((8,8,3),dtype=np.uint8);mask=np.zeros((8,8),dtype=np.int64)
    def model(image):raise AssertionError('must not execute an empty experiment')
    with pytest.raises(ValueError,match='at least one severity'):
        evaluate(model,[image],[mask],1,corruptions=['brightness'],severities=[])


@pytest.mark.parametrize('name',['brightness','elastic_transform'])
@pytest.mark.parametrize('severity',INVALID)
def test_pair_independently_rejects_invalid_severity(name,severity):
    image=np.zeros((8,8,3),dtype=np.uint8);mask=np.zeros((8,8),dtype=np.int64)
    with pytest.raises(ValueError,match='severity must be an integer'):
        corrupt_pair(image,mask,name,severity=severity,seed=0)


def test_fractional_experiment_is_not_silently_a_lower_severity():
    image=np.zeros((8,8,3),dtype=np.uint8);mask=np.zeros((8,8),dtype=np.int64)
    try:
        evaluate(lambda x:mask,[image],[mask],1,corruptions=['brightness'],severities=[1.9])
    except ValueError as error:
        assert 'severity must be an integer' in str(error)
    else:
        raise AssertionError('fractional severity must not produce a different integer experiment')


def test_valid_generator_severities_are_consumed_once_and_preserved():
    image=np.zeros((8,8,3),dtype=np.uint8);mask=np.zeros((8,8),dtype=np.int64)
    result=evaluate(lambda x:mask,[image],[mask],1,corruptions=['brightness'],severities=(s for s in [1,3,5]))
    assert result['severities']==[1,3,5]
    assert list(result['per_corruption']['brightness'])==[1,3,5]


@pytest.mark.parametrize('name',['brightness','elastic_transform'])
def test_rejected_severity_preserves_input_and_rng(name):
    image=np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3)
    mask=np.arange(64,dtype=np.int64).reshape(8,8)%3
    before_image=image.copy();before_mask=mask.copy()
    np.random.seed(786)
    state=np.random.get_state()
    with pytest.raises(ValueError,match='severity must be an integer'):
        corrupt_pair(image,mask,name,severity=1.9,seed=7)
    after=np.random.get_state()
    assert state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:]
    assert np.array_equal(image,before_image) and np.array_equal(mask,before_mask)
