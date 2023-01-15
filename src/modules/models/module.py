from functools import reduce
from typing import Any, List, Optional, Union

import segmentation_models_pytorch as seg_models
import timm
import torch
import torchvision.models as models


def set_parameter_requires_grad(
    model: torch.nn.Module, freeze_layers: Any
) -> None:
    """Freeze layers in PyTorch model by indices.

    Args:
        model (torch.nn.Module): PyTorch model.
        freeze_layers (Any): List of layer indices which should bre frozen.
    """

    if isinstance(freeze_layers, int):
        for i, child in enumerate(model.children()):
            if i <= freeze_layers:
                print(f"Freeze layer: {child}")
                for param in child.parameters():
                    param.requires_grad = False
    elif isinstance(freeze_layers, (list, tuple)):
        for i, child in enumerate(model.children()):
            if i in freeze_layers:
                print(f"Freeze layer: {child}")
                for param in child.parameters():
                    param.requires_grad = False
    else:
        raise ValueError("freeze_layers must be int or list")


def get_module_by_name(
    module: Union[torch.Tensor, torch.nn.Module], access_string: str
) -> Any:
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.

    Args:
        module (Union[torch.Tensor, torch.nn.Module]): Input module.
        access_string (str): Access string.

    Returns:
        Any: Submodule found by access string.
    """

    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def replace_module_by_identity(
    module: torch.nn.Module,
    prev: torch.nn.Module,
    new: torch.nn.Module,
    strategy: str = "id",
) -> None:
    """Replace `prev` submodule to `new` submodule in `module` by exact `object
    id` or `__str__` representation.

    Args:
        module (torch.nn.Module): Input module.
        prev (torch.nn.Module): Submodule which should be replaced.
        new (torch.nn.Module): Submodule which should use for replacing.
        strategy (str): Replacing mode. Could be `id` or `repr`. Default to `id`.
    """
    if strategy not in ("id", "repr"):
        raise ValueError("`strategy` could be `id` or `repr`")

    for name, submodule in module.named_children():
        if len(list(submodule.children())) > 0:
            replace_module_by_identity(submodule, prev, new, strategy)
        if strategy == "id":
            if submodule is prev:
                setattr(module, name, new)
        elif strategy == "repr":
            if str(submodule) == str(prev):
                setattr(module, name, new)


def get_module_attr_by_name_recursively(
    module: torch.nn.Module, index: int, attr_name: str
) -> Any:
    """Get attribute value by name for a module with index `index` across all
    submodules which have attribute with name `attr_name`.

    It doesn't work for non-iterable modules, like ResNet BasicBlock, etc.

    Args:
        module (torch.nn.Module): Input module.
        index (int): Submodule index, 0 for the first and -1 for the last.
        attr_name (str): Submodule attribute name.
    """

    def collect_submodules(submodule: Any):
        if hasattr(submodule, "__getitem__"):
            for _, deeper_submodule in submodule.named_children():
                collect_submodules(deeper_submodule)
            return

        if hasattr(submodule, attr_name):
            submodules_with_attr.append(submodule)

    submodules_with_attr = []
    collect_submodules(module)
    if (
        not submodules_with_attr
        or (len(submodules_with_attr) <= index)
        or not hasattr(submodules_with_attr[index], attr_name)
    ):
        return
    return getattr(submodules_with_attr[index], attr_name)


class BaseModule(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        model_repo: Optional[str] = None,
        freeze_layers: Any = None,
        **kwargs: Any,
    ) -> None:
        """Base Module Wrapper for PyTorch like model.

        Available models registries:

        - torchvision.models
        - segmentation_models_pytorch
        - timm
        - torch.hub

        Args:
            model_name (str): Model name.
            model_repo (:obj:`str`, optional): Model repository.
            freeze_layers (Any): List of layer indices which should bre frozen.
            kwargs (Any): Additional keyword arguments for Model initialization.
        """

        super().__init__()
        if "torchvision.models" in model_name:
            model_name = model_name.split("torchvision.models/")[1]
            self.model = getattr(models, model_name)(**kwargs)
        elif "segmentation_models_pytorch" in model_name:
            model_name = model_name.split("segmentation_models_pytorch/")[1]
            self.model = getattr(seg_models, model_name)(**kwargs)
        elif "timm" in model_name:
            model_name = model_name.split("timm/")[1]
            self.model = timm.create_model(model_name, **kwargs)
        elif "torch.hub" in model_name:
            model_name = model_name.split("torch.hub/")[1]
            if not model_repo:
                raise ValueError("Please provide model_repo for torch.hub")
            self.model = torch.hub.load(model_repo, model_name, **kwargs)
        else:
            raise NotImplementedError(f"Model {model_name} is not implemented")

        if freeze_layers:
            set_parameter_requires_grad(self.model, freeze_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    @staticmethod
    def get_timm_list_models(*args: Any, **kwargs: Any) -> List[str]:
        return timm.list_models(*args, **kwargs)

    @staticmethod
    def get_torch_hub_list_models(
        model_repo: str, *args: Any, **kwargs: Any
    ) -> List[Any]:
        return torch.hub.list(model_repo, *args, **kwargs)

    @staticmethod
    def get_torch_hub_model_help(
        model_repo: str, model_name: str, *args: Any, **kwargs: Any
    ) -> List[Any]:
        return torch.hub.help(model_repo, model_name, *args, **kwargs)
