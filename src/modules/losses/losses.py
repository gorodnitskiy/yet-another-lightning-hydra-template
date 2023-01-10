import hydra
import torch
from omegaconf import DictConfig


def load_loss(loss_cfg: DictConfig) -> torch.nn.Module:
    """Load loss module.

    Args:
        loss_cfg (DictConfig): Loss config.

    Returns:
        torch.nn.Module: Loss module.
    """

    if loss_cfg.get("weight"):
        weight = torch.tensor(loss_cfg.get("weight")).float()
        loss = hydra.utils.instantiate(loss_cfg, weight=weight)
    else:
        loss = hydra.utils.instantiate(loss_cfg)

    return loss
