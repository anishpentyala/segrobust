import numpy as np
import pytest
from segrobust import iou_from_confusion, miou_from_confusion

@pytest.mark.parametrize('function',[iou_from_confusion,miou_from_confusion])
@pytest.mark.parametrize('matrix',[
    [[-1,2],[2,1]], [[1,float('nan')],[0,1]],
    [[1,float('inf')],[0,1]], [[1,2,3],[4,5,6]], [1,2], 1,
])
def test_unusable_confusion_inputs_raise(function,matrix):
    with pytest.raises(ValueError):function(np.asarray(matrix))

@pytest.mark.parametrize('function',[iou_from_confusion,miou_from_confusion])
@pytest.mark.parametrize('matrix',[[['1','0'],['0','1']],[[1j,0],[0,1j]]])
def test_nonreal_storage_is_not_silently_coerced(function,matrix):
    with pytest.raises(TypeError):function(np.asarray(matrix))

@pytest.mark.parametrize('dtype',[np.int64,np.uint64])
def test_large_integer_counts_widen_before_union(dtype):
    cm=np.array([[2**62,0],[0,2**62]],dtype=dtype)
    np.testing.assert_array_equal(iou_from_confusion(cm),[1.,1.])
    assert miou_from_confusion(cm)==1.

def test_large_perfect_counts_do_not_become_nan_control():
    score=miou_from_confusion(np.array([[2**62,0],[0,2**62]],dtype=np.int64))
    assert score==1., 'perfect large integer counts must not wrap into an absent-class score'

def test_negative_counts_do_not_produce_plausible_score_control():
    try:miou_from_confusion(np.array([[-1,2],[2,1]]))
    except ValueError:return
    pytest.fail('negative confusion counts must not produce a segmentation score')

@pytest.mark.parametrize('dtype',[np.bool_,np.uint8,np.int32,np.float32,np.float64])
def test_valid_real_weights_and_counts_stay_supported(dtype):
    matrix=np.eye(2,dtype=dtype)
    np.testing.assert_array_equal(iou_from_confusion(matrix),[1.,1.])
    assert miou_from_confusion(matrix)==1.

def test_fractional_weights_remain_valid():
    np.testing.assert_allclose(iou_from_confusion(np.array([[1.5,.5],[.5,1.5]])),[.6,.6])
    assert miou_from_confusion(np.array([[1.5,.5],[.5,1.5]]))==pytest.approx(.6)

@pytest.mark.parametrize('matrix',[
    [[1e308,0],[0,1e308]], [[1e308,0],[0,1e-308]],
    [[1e-308,0],[0,1e-308]],
])
def test_finite_extreme_weights_stay_perfect(matrix):
    with np.errstate(all='raise'):
        np.testing.assert_array_equal(iou_from_confusion(np.array(matrix)),[1.,1.])
        assert miou_from_confusion(np.array(matrix))==1.

def test_empty_class_table_stays_supported():
    assert iou_from_confusion(np.empty((0,0))).shape==(0,)
    assert np.isnan(miou_from_confusion(np.empty((0,0))))

def test_absent_class_keeps_nan_and_exclusion():
    np.testing.assert_allclose(iou_from_confusion(np.array([[0,0],[0,3]])),[np.nan,1.],equal_nan=True)
    assert miou_from_confusion(np.array([[0,0],[0,3]]))==1.

def test_extreme_finite_weights_do_not_overflow_control():
    score=miou_from_confusion(np.array([[1e308,0],[0,1e308]]))
    assert score==1., 'finite perfect weights must not overflow into a zero score'

def test_per_class_scaling_matches_exact_integer_reference():
    from fractions import Fraction
    rng=np.random.default_rng(815)
    for size in [1,2,3,8]:
        for _ in range(40):
            matrix=rng.integers(0,2**62,size=(size,size),dtype=np.int64)
            raw=matrix.tolist()
            expected=[]
            for i in range(size):
                union=sum(raw[i])+sum(row[i] for row in raw)-raw[i][i]
                expected.append(float(Fraction(raw[i][i],union)) if union else np.nan)
            np.testing.assert_allclose(iou_from_confusion(matrix),expected,rtol=1e-14)

def test_noncontiguous_inputs_are_not_changed():
    original=np.array([[7,1],[2,5]],dtype=np.int64)
    before=original.copy()
    np.testing.assert_allclose(iou_from_confusion(original.T),[.7,.625])
    assert miou_from_confusion(original.T)==pytest.approx(.6625)
    np.testing.assert_array_equal(original,before)

@pytest.mark.skipif(np.finfo(np.longdouble).max<=np.finfo(np.float64).max,
                    reason='longdouble has no extended range on this platform')
def test_extended_finite_real_weights_do_not_narrow_to_infinity():
    cm=np.diag(np.array(['1e400','1e-400'],dtype=np.longdouble))
    np.testing.assert_array_equal(iou_from_confusion(cm),[1.,1.])
    assert miou_from_confusion(cm)==1.

def test_max_finite_mixed_weights_match_scaled_reference():
    largest=np.finfo(np.float64).max
    cm=np.array([[largest,largest],[largest,largest]])
    np.testing.assert_allclose(iou_from_confusion(cm),[1/3,1/3])
    assert miou_from_confusion(cm)==pytest.approx(1/3)
