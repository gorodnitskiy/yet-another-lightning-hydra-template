from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.modules.models.module import BaseModule


class GeM(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = float(p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gem(x, p=self.p, eps=self.eps)

    @staticmethod
    def gem(x: torch.Tensor, p: Any, eps: float) -> torch.Tensor:
        x = x.clamp(min=eps).pow(p)
        return F.avg_pool2d(x, (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(p={self.p:.4f}, eps={str(self.eps)})"
        )


class GeMTrainable(GeM):
    def __init__(self, p: int = 3, eps: float = 1e-6) -> None:
        super().__init__(p, eps)
        self.p = nn.Parameter(torch.ones(1) * torch.tensor(p))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(p={self.p.data.tolist()[0]:.4f}, eps={str(self.eps)})"
        )


class ReIdentificator(BaseModule):
    def __init__(
        self,
        model_name: str,
        head_type: str,
        embedding_size: Optional[int] = None,
        proj_hidden_dim: Optional[int] = None,
        model_repo: Optional[str] = None,
        p: Optional[int] = None,
        gem_trainable: Optional[bool] = None,
        use_pretrained: bool = False,
        freeze_params: Any = None,
        **kwargs,
    ) -> None:
        super().__init__(model_name, model_repo, use_pretrained, freeze_params)
        if head_type == "gem":
            if gem_trainable:
                self.model.avgpool = GeMTrainable(p=p)
            else:
                self.model.avgpool = GeM(p=p)
            self.features_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()
        else:
            assert "resnet18" in model_name
            out_bias = kwargs["bias"] if "bias" in kwargs else False
            self.model.avgpool = nn.Sequential(
                nn.Conv2d(
                    512,
                    512,
                    kernel_size=(3, 5),
                    stride=(1, 1),
                    groups=512,
                    bias=False,
                ),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    512,
                    512,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(512),
                nn.Flatten(),
            )
            self.model.fc = nn.Sequential(
                nn.Linear(proj_hidden_dim, embedding_size, bias=out_bias),
                nn.BatchNorm1d(embedding_size),
            )
            self.features_dim = embedding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
