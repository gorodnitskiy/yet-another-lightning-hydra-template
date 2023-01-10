from typing import Any, List, Optional

import segmentation_models_pytorch as seg_models
import timm
import torch
import torchvision.models as models


def set_parameter_requires_grad(
    model: torch.nn.Module, freeze_params: Any
) -> None:
    if isinstance(freeze_params, int):
        for i, child in enumerate(model.children()):
            if i <= freeze_params:
                print(f"Freeze layer: {child}")
                for param in child.parameters():
                    param.requires_grad = False
    elif isinstance(freeze_params, (list, tuple)):
        for i, child in enumerate(model.children()):
            if i in freeze_params:
                print(f"Freeze layer: {child}")
                for param in child.parameters():
                    param.requires_grad = False
    else:
        raise ValueError("freeze_params must be int or list")


class BaseModule(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        model_repo: Optional[str] = None,
        use_pretrained: bool = True,
        freeze_params: Any = None,
        **kwargs: Any,
    ) -> None:
        """Available models registries:

        - torchvision.models
        - segmentation_models_pytorch
        - timm
        - torch.hub
        """
        super().__init__()
        if "torchvision.models" in model_name:
            model_name = model_name.split("torchvision.models/")[1]
            self.model = getattr(models, model_name)(
                pretrained=use_pretrained, **kwargs
            )
        elif "segmentation_models_pytorch" in model_name:
            model_name = model_name.split("segmentation_models_pytorch/")[1]
            self.model = getattr(seg_models, model_name)(
                pretrained=use_pretrained, **kwargs
            )
        elif "timm" in model_name:
            model_name = model_name.split("timm/")[1]
            self.model = timm.create_model(
                model_name, pretrained=use_pretrained, **kwargs
            )
        elif "torch.hub" in model_name:
            model_name = model_name.split("torch.hub/")[1]
            if not model_repo:
                raise ValueError("Please provide model_repo for torch.hub")
            self.model = torch.hub.load(
                model_repo, model_name, pretrained=use_pretrained, **kwargs
            )
        else:
            raise NotImplementedError(f"Model {model_name} is not implemented")

        if freeze_params:
            set_parameter_requires_grad(self.model, freeze_params)

    @staticmethod
    def get_timm_list_models(*args: Any, **kwargs: Any) -> List[str]:
        return timm.list_models(*args, **kwargs)

    @staticmethod
    def get_torch_hub_list_models(
        model_repo: str, *args: Any, **kwargs: Any
    ) -> List[Any]:
        """
        github: (str) – a string with format <repo_owner/repo_name
        """
        return torch.hub.list(model_repo, *args, **kwargs)

    @staticmethod
    def get_torch_hub_model_help(
        model_repo: str, model_name: str, *args: Any, **kwargs: Any
    ) -> List[Any]:
        """
        github: (str) – a string with format <repo_owner/repo_name
        """
        return torch.hub.help(model_repo, model_name, *args, **kwargs)
