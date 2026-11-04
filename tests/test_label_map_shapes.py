"""Evaluation must not flatten a channel axis into a plausible segmentation score."""
import numpy as np
import pytest
from segrobust import corrupt_pair, evaluate


@pytest.mark.parametrize('shape', [(8,8,1),(8,8,3),(8,8,0),(8,),(64,),(),(4,16)])
@pytest.mark.parametrize('name',['brightness','elastic_transform'])
def test_pair_rejects_non_label_map_shape(shape,name):
    im=np.zeros((8,8,3),dtype=np.uint8);mask=np.zeros(shape,dtype=np.int64)
    with pytest.raises(ValueError,match='mask must be a 2D ndarray'):
        corrupt_pair(im,mask,name,1,seed=0)


@pytest.mark.parametrize('dtype',[float,complex,object,'U1'])
@pytest.mark.parametrize('name',['brightness','elastic_transform'])
def test_pair_rejects_noninteger_mask_before_corruption(dtype,name):
    im=np.zeros((8,8,3),dtype=np.uint8)
    with pytest.raises(TypeError,match='mask labels must have integer'):
        corrupt_pair(im,np.zeros((8,8),dtype=dtype),name,1,seed=0)


@pytest.mark.parametrize('shape',[(8,8,1),(8,8,3),(8,),(4,16)])
def test_complete_dataset_shape_validation_precedes_model_and_progress(shape):
    im=np.zeros((8,8,3),dtype=np.uint8);valid=np.zeros((8,8),dtype=np.int64);bad=np.zeros(shape,dtype=np.int64)
    calls=[]
    def model(im):calls.append('model');return valid
    with pytest.raises(ValueError,match='mask must be a 2D ndarray'):
        evaluate(model,[im,im],[valid,bad],1,corruptions=[],progress=calls.append)
    assert calls==[]


@pytest.mark.parametrize('shape',[(8,8,1),(8,8,3),(8,),(4,16)])
def test_model_predictions_require_the_exact_two_dimensional_label_map(shape):
    im=np.zeros((8,8,3),dtype=np.uint8);mask=np.zeros((8,8),dtype=np.int64)
    with pytest.raises(ValueError,match='model_fn returned shape'):
        evaluate(lambda _:np.zeros(shape,dtype=np.int64),[im],[mask],1,corruptions=[])


def test_color_channel_mask_cannot_be_reported_as_a_perfect_segmentation_control():
    im=np.zeros((8,8,3),dtype=np.uint8);mask=np.zeros((8,8,3),dtype=np.int64)
    try:
        evaluate(lambda _:mask,[im],[mask],1,corruptions=['brightness'],severities=[1])
    except ValueError as e:
        assert 'mask must be a 2D ndarray' in str(e)
        return
    raise AssertionError('a channel-axis mask must not become a perfect segmentation score')


@pytest.mark.parametrize('dtype',[np.bool_,np.uint8,np.int8,np.int32,np.uint64,np.int64])
@pytest.mark.parametrize('shape',[(8,8),(8,8,1),(8,8,3)])
def test_valid_label_map_and_image_layout_keep_perfect_score(dtype,shape):
    im=np.zeros(shape,dtype=np.uint8);mask=np.zeros((8,8),dtype=dtype)
    result=evaluate(lambda _:mask,[im],[mask],1,corruptions=['brightness'],severities=[1])
    assert result['clean_miou']==result['mean_corrupt_miou']==1.0


def test_rejected_mask_preserves_rng_and_image():
    im=np.full((8,8,3),128,dtype=np.uint8);before=im.copy()
    mask=np.zeros((8,8,3),dtype=np.int64)
    np.random.seed(21);state=np.random.get_state()
    with pytest.raises(ValueError):corrupt_pair(im,mask,'elastic_transform',3,seed=0)
    after=np.random.get_state()
    assert state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:]
    assert np.array_equal(im,before)


@pytest.mark.parametrize('dtype',[float,complex,object,'U1'])
def test_complete_dataset_integer_validation_precedes_model_and_progress(dtype):
    im=np.zeros((8,8,3),dtype=np.uint8);valid=np.zeros((8,8),dtype=np.int64);bad=np.zeros((8,8),dtype=dtype)
    calls=[]
    def model(im):calls.append('model');return valid
    with pytest.raises(TypeError,match='mask labels must have integer'):
        evaluate(model,[im,im],[valid,bad],1,corruptions=[],progress=calls.append)
    assert calls==[]


@pytest.mark.parametrize('bad',[None, [[0]*8]*8, np.zeros((8,8),dtype=np.float32),np.zeros((4,8,3),dtype=np.uint8)])
def test_complete_dataset_image_validation_precedes_model_and_progress(bad):
    im=np.zeros((8,8,3),dtype=np.uint8);mask=np.zeros((8,8),dtype=np.int64)
    calls=[]
    def model(im):calls.append('model');return mask
    with pytest.raises((TypeError,ValueError)):
        evaluate(model,[im,bad],[mask,mask],1,corruptions=[],progress=calls.append)
    assert calls==[]


@pytest.mark.parametrize('name',['brightness','elastic_transform'])
def test_non_array_image_pair_is_a_validation_error(name):
    with pytest.raises(TypeError,match='image must be a numpy.ndarray'):
        corrupt_pair(None,np.zeros((8,8),dtype=np.int64),name)
