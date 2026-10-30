"""Composable classification-versus-detection task loss weighting."""
import torch
from torch import nn


class UncertaintyTaskWeights(nn.Module):
    """Homoscedastic uncertainty weighting for the two task losses (Kendall et al., CVPR 2018)."""

    def __init__(self, initial_log_variance=0.0):
        super().__init__()
        self.log_variances = nn.Parameter(torch.full((2,), float(initial_log_variance)))

    def forward(self, classification_loss, detection_loss):
        losses = torch.stack((classification_loss, detection_loss))
        precision = torch.exp(-self.log_variances)
        return (precision * losses + self.log_variances).sum(), {
            'classification': float(precision[0].detach()),
            'detection': float(precision[1].detach()),
        }


def combine_task_losses(classification_loss, detector_losses, classification_weight=1.0,
                        detection_weight=1.0, uncertainty=None):
    """Keep the detector's four terms as one task and expose component logs to the caller."""
    detection_loss = sum(detector_losses.values())
    if uncertainty is not None:
        total, effective_weights = uncertainty(classification_loss, detection_loss)
    else:
        total = classification_weight * classification_loss + detection_weight * detection_loss
        effective_weights = {'classification': classification_weight, 'detection': detection_weight}
    return total, detection_loss, effective_weights
