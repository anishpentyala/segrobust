"""Arithmetic uses index width, not the storage width of label images."""
import numpy as np
import pytest
from segrobust import confusion_matrix, miou_from_confusion, evaluate


@pytest.mark.parametrize('dtype,n', [(np.uint8,21),(np.int8,21),(np.uint8,256),(np.uint16,300),(np.int16,200)])
def test_perfect_labels_keep_exact_bins(dtype,n):
    labels=np.arange(n,dtype=dtype)
    cm=confusion_matrix(labels,labels,n)
    np.testing.assert_array_equal(cm,np.eye(n,dtype=np.int64))
    assert miou_from_confusion(cm)==1.0


@pytest.mark.parametrize('pd,td', [(np.uint8,np.uint8),(np.int8,np.uint8),(np.uint64,np.int64),(np.int64,np.uint64)])
def test_off_diagonal_counts_match_wide_reference(pd,td):
    pred=np.array([20,0,19,4,2,20],dtype=pd)
    target=np.array([20,20,0,19,2,20],dtype=td)
    actual=confusion_matrix(pred,target,21)
    expected=np.zeros((21,21),dtype=np.int64)
    for t,p in zip(target.tolist(),pred.tolist()): expected[t,p]+=1
    np.testing.assert_array_equal(actual,expected)


def test_evaluate_perfect_uint8_twenty_one_classes():
    mask=np.arange(64,dtype=np.uint8).reshape(8,8)%21
    image=np.repeat(mask[:,:,None],3,axis=2)
    result=evaluate(lambda image:mask.copy(),[image],[mask],21,corruptions=['brightness'],severities=[1])
    assert result['clean_miou']==1.0, 'perfect uint8 labels must score clean mIoU 1.0'
    assert result['per_corruption']['brightness'][1]==1.0


def test_ignored_uint64_sentinel_not_cast_before_filter():
    maximum=np.iinfo(np.uint64).max
    p=np.array([20,maximum],dtype=np.uint64)
    t=np.array([20,maximum],dtype=np.uint64)
    cm=confusion_matrix(p,t,21,ignore_index=maximum)
    assert cm[20,20]==1 and cm.sum()==1


@pytest.mark.parametrize('which',['pred','target'])
@pytest.mark.parametrize('value',[0.0,0.5,float('nan'),float('inf')])
def test_noninteger_storage_is_not_silently_truncated(which,value):
    kwargs={'pred':np.array([0]),'target':np.array([0])}
    kwargs[which]=np.array([value])
    with pytest.raises((TypeError,ValueError)):
        confusion_matrix(**kwargs,num_classes=21)


@pytest.mark.parametrize('dtype',[np.uint8,np.uint16,np.uint32,np.uint64,np.int8,np.int16,np.int32,np.int64])
def test_repeated_valid_narrow_counts_and_ignored_values(dtype):
    labels=np.array([0,20,20,2,0],dtype=dtype)
    predictions=np.array([0,19,20,2,20],dtype=dtype)
    cm=confusion_matrix(predictions,labels,21,ignore_index=2)
    assert cm.sum()==4 and cm[0,0]==1 and cm[0,20]==1 and cm[20,19]==1 and cm[20,20]==1


def test_out_of_range_uint64_prediction_still_rejected():
    with pytest.raises(ValueError,match='out of range'):
        confusion_matrix(np.array([np.iinfo(np.uint64).max],dtype=np.uint64),np.array([20],dtype=np.uint8),21)


@pytest.mark.parametrize('n',[0,-1,True,np.bool_(False),2.5,'21',np.iinfo(np.intp).max])
def test_invalid_class_count_cannot_create_invalid_bin_indices(n):
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0],dtype=np.uint8),np.array([0],dtype=np.uint8),n)


def test_numpy_class_count_and_boolean_binary_labels_remain_supported():
    cm=confusion_matrix(np.array([False,True]),np.array([False,True]),np.int64(2))
    np.testing.assert_array_equal(cm,np.eye(2,dtype=np.int64))


def test_random_histograms_match_python_integer_reference():
    rng=np.random.default_rng(17)
    for n in (2,21,127,256,300):
        for dtype in (np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64):
            if n-1>np.iinfo(dtype).max:
                continue
            target=rng.integers(0,n,size=1000).astype(dtype)
            pred=rng.integers(0,n,size=1000).astype(dtype)
            expected=np.zeros((n,n),dtype=np.int64)
            for t,p in zip(target.tolist(),pred.tolist()): expected[t,p]+=1
            np.testing.assert_array_equal(confusion_matrix(pred,target,n),expected)
