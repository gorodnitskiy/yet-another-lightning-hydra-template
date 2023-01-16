from typing import Any, Dict

import omegaconf
import pytest

from src.modules.losses import load_loss

_TEST_LOSS_CFG = (
    {
        "_target_": "src.modules.losses.AngularPenaltySMLoss",
        "embedding_size": 16,
        "num_classes": 10,
    },
    {"_target_": "src.modules.losses.FocalLoss"},
    {
        "_target_": "segmentation_models_pytorch.losses.DiceLoss",
        "mode": "binary",
    },
    {
        "_target_": "segmentation_models_pytorch.losses.FocalLoss",
        "mode": "binary",
    },
    {
        "_target_": "segmentation_models_pytorch.losses.JaccardLoss",
        "mode": "binary",
    },
    {
        "_target_": "segmentation_models_pytorch.losses.LovaszLoss",
        "mode": "binary",
    },
    {"_target_": "torch.nn.BCEWithLogitsLoss"},
    {"_target_": "torch.nn.BCEWithLogitsLoss", "pos_weight": (2.0,)},
    {"_target_": "torch.nn.BCEWithLogitsLoss", "pos_weight": (0.5,)},
    {"_target_": "torch.nn.BCELoss"},
    {"_target_": "torch.nn.CrossEntropyLoss"},
    {"_target_": "torch.nn.CrossEntropyLoss", "weight": (0.1, 0.5, 0.4)},
    {"_target_": "torch.nn.CrossEntropyLoss", "weight": (1.0, 5.0, 4.0)},
    {"_target_": "torch.nn.MSELoss"},
    {"_target_": "torch.nn.SmoothL1Loss"},
    {"_target_": "src.modules.losses.VicRegLoss"},
)


@pytest.mark.parametrize("loss_cfg", _TEST_LOSS_CFG)
def test_loss_cfg(loss_cfg: Dict[str, Any]):
    cfg = omegaconf.OmegaConf.create(loss_cfg)
    _ = load_loss(cfg)
