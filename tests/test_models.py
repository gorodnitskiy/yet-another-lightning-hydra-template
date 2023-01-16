from typing import Any, Dict, List

import hydra
import omegaconf
import pytest
import torch

from src.modules.models.module import (
    get_module_attr_by_name_recursively,
    get_module_by_name,
    replace_module_by_identity,
)

_MODULE_SOURCE = (
    {"model_name": "torchvision.models/resnet18", "weights": None},
    {"model_name": "timm/tf_efficientnetv2_s", "pretrained": False},
    {
        "model_name": "segmentation_models_pytorch/DeepLabV3Plus",
        "encoder_name": "resnet34",
        "encoder_weights": None,
        "in_channels": 3,
        "classes": 10,
    },
    # In pytest torch.hub.load does not work correct when torchvision is installed
    # https://discuss.pytorch.org/t/problem-with-loading-models-from-torch-hub-pytorch-vision-in-pytest/170320
    # {
    #     "model_name": "torch.hub/regnet_x_400mf",
    #     "model_repo": "pytorch/vision",
    #     "weights": None,
    # },
)

_REID_PARAMS = (
    {
        "head_type": "fc",
        "embedding_size": 128,
        "proj_hidden_dim": 2880,
        "kernel_size": [5, 7],
    },
    {"head_type": "gem", "p": 3},
)


@pytest.mark.parametrize("model_source_params", _MODULE_SOURCE)
def test_base_module(model_source_params: Dict[str, Any]):
    cfg = {
        "_target_": "src.modules.models.module.BaseModule",
        **model_source_params,
    }
    cfg = omegaconf.OmegaConf.create(cfg)
    model = hydra.utils.instantiate(cfg)
    model.eval()

    tensor = torch.randn((2, 3, 224, 224))
    _ = model.forward(tensor)


@pytest.mark.parametrize("num_classes", [1, 10])
def test_classifier(num_classes: int):
    cfg = {
        "_target_": "src.modules.models.classification.Classifier",
        "model_name": "torchvision.models/mobilenet_v3_large",
        "weights": None,
        "num_classes": num_classes,
    }
    cfg = omegaconf.OmegaConf.create(cfg)
    model = hydra.utils.instantiate(cfg)
    model.eval()

    tensor = torch.randn((2, 3, 224, 224))
    output = model.forward(tensor)

    assert output.size(dim=0) == 2
    if num_classes == 1:
        assert len(output.size()) == 1
    else:
        assert output.size(dim=1) == num_classes


@pytest.mark.parametrize("num_classes", [[92, 4, 7], [1, 4], [10]])
def test_classifier_multiple_head(num_classes: List[int]):
    cfg = {
        "_target_": "src.modules.models.classification.ClassifierMultipleHead",
        "model_name": "torchvision.models/resnext50_32x4d",
        "weights": None,
        "num_classes": num_classes,
    }
    cfg = omegaconf.OmegaConf.create(cfg)
    model = hydra.utils.instantiate(cfg)
    model.eval()

    tensor = torch.randn((2, 3, 224, 224))
    outputs = model.forward(tensor)

    assert all([output.size(dim=0) == 2 for output in outputs])
    statements = []
    for nums, output in zip(num_classes, outputs):
        if nums == 1:
            statements.append(len(output.size()) == 1)
        else:
            statements.append(output.size(dim=1) == nums)
    assert all(statements)


def test_backbone_vicreg(model_name: str = "resnet18"):
    cfg = {
        "_target_": "src.modules.models.classification.BackboneVicReg",
        "model_name": f"torchvision.models/{model_name}",
        "weights": None,
    }
    cfg = omegaconf.OmegaConf.create(cfg)
    model = hydra.utils.instantiate(cfg)
    model.eval()

    tensor = torch.randn((2, 3, 224, 224))
    _ = model.forward(tensor)


@pytest.mark.parametrize("params", _REID_PARAMS)
def test_reidentificator(params: Dict[str, Any]):
    cfg = {
        "_target_": "src.modules.models.reidentification.ReIdentificator",
        "model_name": "torchvision.models/mobilenet_v3_large",
        "weights": None,
        **params,
    }
    cfg = omegaconf.OmegaConf.create(cfg)
    model = hydra.utils.instantiate(cfg)
    model.eval()

    tensor = torch.randn((2, 3, 224, 224))
    _ = model.forward(tensor)


@pytest.mark.parametrize("model_name", ["mobilenet_v3_large", "vit_b_16"])
def test_module_utils(model_name: str):
    cfg = {
        "_target_": "src.modules.models.module.BaseModule",
        "model_name": f"torchvision.models/{model_name}",
        "weights": None,
    }
    cfg = omegaconf.OmegaConf.create(cfg)
    model = hydra.utils.instantiate(cfg).model
    submodules = [name for name, _ in model.named_children()]

    head = get_module_by_name(model, submodules[-1])
    encoder = get_module_by_name(model, submodules[-3])

    head_in_features = get_module_attr_by_name_recursively(
        head, 0, "in_features"
    )
    encoder_out_channels = get_module_attr_by_name_recursively(
        encoder, -1, "out_channels"
    )

    assert head_in_features == encoder_out_channels

    replace_module_by_identity(
        model,
        head,
        torch.nn.Linear(in_features=head_in_features, out_features=10),
    )
    new_head = get_module_by_name(
        model, [name for name, _ in model.named_children()][-1]
    )
    new_head_in_features = get_module_attr_by_name_recursively(
        new_head, 0, "in_features"
    )
    new_head_out_features = get_module_attr_by_name_recursively(
        new_head, -1, "out_features"
    )

    assert new_head_in_features == head_in_features
    assert new_head_out_features == 10
