import pytest
from torch import nn

from sarbench.compliance import parameter_counts, validate_input_size, validate_model_budget


def test_parameter_budget_counts_frozen_and_trainable_parameters():
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    model[0].requires_grad_(False)
    assert parameter_counts(model) == {'total': 23, 'trainable': 8}
    assert validate_model_budget(model) == {'total': 23, 'trainable': 8}
    with pytest.raises(ValueError):
        validate_model_budget(model, total_limit=20)


@pytest.mark.parametrize('height,width', [(256, 256), (512, 512), (256, 384)])
def test_native_sizes_are_legal(height, width):
    validate_input_size(height, width)


def test_upscaled_or_oversized_inputs_are_rejected():
    with pytest.raises(ValueError):
        validate_input_size(513, 512)
