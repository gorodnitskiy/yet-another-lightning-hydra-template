from typing import Any, List, Optional

import torch

from src.modules.models.module import BaseModule


class ConvActLin(torch.nn.Module):
    def __init__(
        self,
        in_channels_2d,
        out_channels_2d,
        num_classes,
        kernel_size_2d=(1, 1),
        activation=torch.nn.ReLU,
        **kwargs,
    ) -> None:
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels_2d, out_channels_2d, kernel_size_2d, **kwargs
            ),
            activation(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(out_channels_2d, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1).unsqueeze(-1)
        return self.head(x)


class Classifier(BaseModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        use_pretrained: bool = True,
        model_repo: Optional[str] = None,
        freeze_params: Any = None,
        **kwargs,
    ) -> None:
        super().__init__(model_name, model_repo, use_pretrained, freeze_params)
        bias = kwargs["bias"] if "bias" in kwargs else False
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(
            in_features, num_classes, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class ClassifierMultipleHead(BaseModule):
    def __init__(
        self,
        model_name: str,
        num_classes: List[int],
        use_pretrained: bool = True,
        model_repo: Optional[str] = None,
        freeze_params: Any = None,
        **kwargs,
    ) -> None:
        super().__init__(model_name, model_repo, use_pretrained, freeze_params)
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Identity()
        self.heads = []
        for num_class in num_classes:
            self.heads.append(
                ConvActLin(
                    in_features, in_features, num_class, **kwargs
                )
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.model(x)
        output = []
        for head in self.heads:
            output.append(head(x))
        return output
