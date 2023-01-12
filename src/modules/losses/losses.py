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

    weight_params = {}
    for param_name, param_value in loss_cfg.items():
        if "weight" in param_name:
            weight_params[param_name] = torch.tensor(param_value).float()

    loss = hydra.utils.instantiate(loss_cfg, **weight_params)

    return loss
