import omegaconf
import pytest

from src.modules.losses.losses import load_loss

_IMPLEMENTED_LOSS_NAMES = (
    "AngularPenaltySMLoss",
    "FocalLossManual",
    "segmentation_models_pytorch/DiceLoss",
    "segmentation_models_pytorch/FocalLoss",
    "segmentation_models_pytorch/JaccardLoss",
    "segmentation_models_pytorch/LovaszLoss",
    "segmentation_models_pytorch/TverskyLoss",
    "torch.nn/BCEWithLogitsLoss",
    "torch.nn/BCELoss",
    "torch.nn/CrossEntropyLoss",
    "torch.nn/L1Loss",
    "torch.nn/MSELoss",
    "torch.nn/NLLLoss",
    "torch.nn/SmoothL1Loss",
    "VicRegLoss",
)


@pytest.mark.parametrize("loss_name", _IMPLEMENTED_LOSS_NAMES)
def test_loss(loss_name: str):
    cfg = {"weighted": False, "name": loss_name, "params": None}
    if "segmentation_models_pytorch/" in loss_name:
        cfg.update({"params": {"mode": "binary"}})
    elif loss_name == "AngularPenaltySMLoss":
        cfg.update({"params": {"embedding_size": 16, "num_classes": 10}})
    cfg = omegaconf.OmegaConf.create(cfg)
    _ = load_loss(cfg)
