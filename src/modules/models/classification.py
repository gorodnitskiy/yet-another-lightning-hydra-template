from typing import Any, List, Optional

import torch
from torch import nn

from src.modules.models.module import (
    BaseModule,
    get_module_attr_by_name_recursively,
    get_module_by_name,
    replace_module_by_identity,
)


class ConvActLin(nn.Module):
    def __init__(
        self,
        in_channels_2d,
        out_channels_2d,
        num_classes,
        kernel_size_2d=(1, 1),
        activation=nn.ReLU,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels_2d, out_channels_2d, kernel_size_2d, **kwargs
            ),
            activation(inplace=True),
            nn.Flatten(),
            nn.Linear(out_channels_2d, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1).unsqueeze(-1)
        return self.head(x)


class Classifier(BaseModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        model_repo: Optional[str] = None,
        freeze_layers: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, model_repo, freeze_layers, **kwargs)
        # get head module
        head = get_module_by_name(
            self.model, [name for name, _ in self.model.named_children()][-1]
        )
        # get in_features to head module
        in_features = get_module_attr_by_name_recursively(
            head, 0, "in_features"
        )
        # replace head module to new module
        replace_module_by_identity(
            self.model, head, nn.Linear(in_features, num_classes, bias=True)
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        if self.num_classes == 1:
            x = x.squeeze(dim=1)
        return x


class ClassifierMultipleHead(BaseModule):
    def __init__(
        self,
        model_name: str,
        num_classes: List[int],
        model_repo: Optional[str] = None,
        freeze_layers: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, model_repo, freeze_layers, **kwargs)
        head = get_module_by_name(
            self.model, [name for name, _ in self.model.named_children()][-1]
        )
        in_features = get_module_attr_by_name_recursively(
            head, 0, "in_features"
        )
        replace_module_by_identity(self.model, head, nn.Identity())
        heads = []
        for num_class in num_classes:
            heads.append(ConvActLin(in_features, in_features, num_class))
        self.num_classes = num_classes
        self.heads = nn.ModuleList(heads)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.model(x)
        outputs = []
        for head, num_classes in zip(self.heads, self.num_classes):
            output = head(x)
            if num_classes == 1:
                output = output.squeeze(dim=1)
            outputs.append(output)
        return outputs


class BackboneVicReg(BaseModule):
    def __init__(
        self,
        model_name: str,
        model_repo: Optional[str] = None,
        freeze_layers: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, model_repo, freeze_layers, **kwargs)
        head = get_module_by_name(
            self.model, [name for name, _ in self.model.named_children()][-1]
        )
        replace_module_by_identity(self.model, head, nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
