import torch
from torch.nn import functional as fn


def reduce(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduces the given tensor using a specific criterion.

    Args:
        tensor (torch.Tensor): input tensor
        reduction (str): string with fixed values [elementwise_mean, none, sum]
    Raises:
        ValueError: when the reduction is not supported
    Returns:
        torch.Tensor: reduced tensor, or the tensor itself
    """
    if reduction in ("elementwise_mean", "mean"):
        return torch.mean(tensor)
    elif reduction == "sum":
        return torch.sum(tensor)
    elif reduction is None or reduction == "none":
        return tensor
    raise ValueError("Reduction parameter unknown.")


class FocalLossManual(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = fn.cross_entropy(
            inputs, targets, reduction="none", ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        return reduce(focal_loss, reduction=self.reduction)
