"""Never score underlying invalid pixels by silently dropping NumPy masks."""
import numpy as np
import pytest
from segrobust import corrupt,corrupt_pair,evaluate
from segrobust.metrics import confusion_matrix,iou_from_confusion,miou_from_confusion


def array(data,mask):
    return np.ma.array(data,mask=mask)


@pytest.mark.parametrize('which',['pred','target','both'])
@pytest.mark.parametrize('mask',[False,True,[False,True]])
def test_confusion_rejects_masked_labels(which,mask):
    pred=np.array([0,1]);target=np.array([0,0])
    if which in ('pred','both'):pred=array(pred,mask)
    if which in ('target','both'):target=array(target,mask)
    with pytest.raises(TypeError,match='masked label arrays'):
        confusion_matrix(pred,target,2,ignore_index=255)


@pytest.mark.parametrize('function',[iou_from_confusion,miou_from_confusion])
@pytest.mark.parametrize('mask',[False,True,[[False,True],[False,False]]])
def test_masked_count_table_is_not_treated_as_complete(function,mask):
    cm=array([[1,100],[0,1]],mask)
    with pytest.raises(TypeError,match='masked confusion matrices'):
        function(cm)


@pytest.mark.parametrize('shape',[(8,8),(8,8,1),(8,8,3)])
@pytest.mark.parametrize('mask',[False,True])
def test_corrupt_rejects_masked_image(shape,mask):
    image=array(np.ones(shape,dtype=np.uint8)*100,mask)
    with pytest.raises(TypeError,match='masked image'):
        corrupt(image,'gaussian_noise',1,seed=0)


@pytest.mark.parametrize('geometric',[False,True])
@pytest.mark.parametrize('mask',[False,True])
def test_pair_rejects_masked_labels_before_transform(geometric,mask):
    image=np.ones((8,8,3),dtype=np.uint8)*100
    labels=array(np.zeros((8,8),dtype=np.uint8),mask)
    with pytest.raises(TypeError,match='masked label'):
        corrupt_pair(image,labels,'elastic_transform' if geometric else 'gaussian_noise',1,seed=0)


@pytest.mark.parametrize('which',['image','target'])
@pytest.mark.parametrize('position',[0,1])
def test_evaluate_preflight_rejects_whole_dataset_before_callbacks(which,position):
    images=[np.ones((8,8,3),dtype=np.uint8)*100 for _ in range(2)]
    targets=[np.zeros((8,8),dtype=np.uint8) for _ in range(2)]
    values=images if which=='image' else targets
    values[position]=array(values[position],True)
    def forbidden(*args):pytest.fail('preflight must reject before model or progress')
    with pytest.raises(TypeError,match='masked'):
        evaluate(forbidden,images,targets,2,corruptions=['gaussian_noise'],severities=[1],progress=forbidden)


@pytest.mark.parametrize('masked_call',[1,2])
@pytest.mark.parametrize('include',[False,True])
def test_evaluation_rejects_masked_predictions_in_clean_or_corrupt_pass(masked_call,include):
    calls=[]
    def model(image):
        calls.append(1)
        pred=np.zeros((8,8),dtype=np.uint8)
        return array(pred,True) if len(calls)==masked_call else pred
    with pytest.raises(TypeError,match='masked model predictions'):
        evaluate(model,[np.zeros((8,8,3),dtype=np.uint8)],[np.zeros((8,8),dtype=np.uint8)],2,
                 corruptions=['gaussian_noise'],severities=[1],include_per_class=include)
    assert len(calls)==masked_call


def test_explicit_ignore_conversion_preserves_intended_denominator():
    target=array([0,0],mask=[False,True])
    pred=np.array([0,1])
    cm=confusion_matrix(pred,target.filled(255),2,ignore_index=255)
    assert cm.tolist()==[[1,0],[0,0]]
    assert miou_from_confusion(cm)==1.0
    # Explicitly choosing underlying data instead includes that pixel.
    raw=confusion_matrix(pred,target.data,2)
    assert raw.tolist()==[[1,1],[0,0]]
    assert miou_from_confusion(raw)==0.5
