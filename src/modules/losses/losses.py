from typing import Any

from torch import nn
from omegaconf import DictConfig
from segmentation_models_pytorch import losses

from src.modules.losses.components.focal_loss import FocalLossManual
from src.modules.losses.components.margin_loss import AngularPenaltySMLoss
from src.modules.losses.components.vicreg_loss import VicRegLoss


def load_loss(config: DictConfig, datamodule: Any = None) -> nn.Module:
    weights = None
    if config.get("weighted"):
        if not config.get("split"):
            raise RuntimeError("Weighted Loss requires split name.")
        if config.get("dataset"):
            weights = datamodule.get_weights(
                split_name=config.get("split"),
                dataset_name=config.get("dataset"),
            )
        else:
            weights = datamodule.get_weights(split_name=config.get("split"))

    params = {}
    if config.get("params"):
        params = config.params
    if "torch.nn" in config.name:
        name = config.name.split("torch.nn/")[1]
        if hasattr(getattr(nn, name), "weight"):
            loss = getattr(nn, name)(weight=weights, **params)
        else:
            loss = getattr(nn, name)(**params)
    elif config.name == "VicRegLoss":
        loss = VicRegLoss(**params)
    elif config.name == "AngularPenaltySMLoss":
        loss = AngularPenaltySMLoss(**params)
    elif config.name == "FocalLossManual":
        loss = FocalLossManual(**params)
    elif "segmentation_models_pytorch" in config.name:
        name = config.name.split("segmentation_models_pytorch/")[1]
        loss = getattr(losses, name)(**params)
    else:
        raise NotImplementedError(f"{config.name} loss isn't implemented.")

    return loss
