"""Assignment constraint checks that can be called before a future training run."""

TOTAL_PARAMETER_LIMIT = 130_000_000
TRAINABLE_PARAMETER_LIMIT = 40_000_000
MAX_INPUT_SIZE = 512


def parameter_counts(model):
    """Return total and trainable parameter counts without running a model forward pass."""
    return {
        'total': sum(parameter.numel() for parameter in model.parameters()),
        'trainable': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def validate_model_budget(model, total_limit=TOTAL_PARAMETER_LIMIT,
                          trainable_limit=TRAINABLE_PARAMETER_LIMIT):
    counts = parameter_counts(model)
    if counts['total'] > total_limit or counts['trainable'] > trainable_limit:
        raise ValueError(
            f"Model exceeds assignment limits: {counts['total']:,} total / "
            f"{counts['trainable']:,} trainable; limits are {total_limit:,} / {trainable_limit:,}"
        )
    return counts


def validate_input_size(height, width):
    if height <= 0 or width <= 0 or max(height, width) > MAX_INPUT_SIZE:
        raise ValueError(f'Input must have positive dimensions no larger than {MAX_INPUT_SIZE}px')
