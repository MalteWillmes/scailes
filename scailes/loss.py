import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss


class FocalLoss(nn.Module):
    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        """
        Wrapper for the focal loss function.

        Args:
            alpha (float): Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default: 0.25.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples. Default: 2.
            reduction (str): 'none' | 'mean' | 'sum'
                    'none': no reduction will be applied,
                    'mean': the sum of the output will be divided by the number of
                    elements in the output,
                    'sum': the output will be summed. Default: 'mean'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the focal loss.
        """
        targets = F.one_hot(targets, num_classes=2).to(torch.float)
        return sigmoid_focal_loss(
            inputs,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
