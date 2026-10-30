import torch

from sarbench.losses import UncertaintyTaskWeights, combine_task_losses


def test_uncertainty_weights_combine_classification_and_detection_as_two_tasks():
    weighting = UncertaintyTaskWeights()
    classification = torch.tensor(1.25, requires_grad=True)
    detector = {'rpn_objectness': torch.tensor(0.5), 'roi_classification': torch.tensor(0.75)}
    total, detection_sum, coefficients = combine_task_losses(classification, detector, uncertainty=weighting)
    total.backward()
    assert total.isfinite() and detection_sum.item() == 1.25
    assert weighting.log_variances.grad is not None
    assert coefficients == {'classification': 1.0, 'detection': 1.0}


def test_fixed_task_weights_preserve_component_sum():
    total, det, coefficients = combine_task_losses(
        torch.tensor(2.0), {'rpn': torch.tensor(1.0), 'roi': torch.tensor(3.0)}, 0.5, 1.0)
    assert total.item() == 5.0 and det.item() == 4.0
    assert coefficients == {'classification': 0.5, 'detection': 1.0}
