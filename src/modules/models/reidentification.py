from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.modules.models.module import (
    BaseModule,
    get_module_attr_by_name_recursively,
    get_module_by_name,
    replace_module_by_identity,
)


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
        kernel_size: Optional[Tuple[int, int]] = None,
        proj_hidden_dim: Optional[int] = None,
        model_repo: Optional[str] = None,
        p: Optional[int] = None,
        gem_trainable: bool = False,
        freeze_layers: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, model_repo, freeze_layers, **kwargs)
        head = get_module_by_name(
            self.model, [name for name, _ in self.model.named_children()][-1]
        )
        avg_pool = get_module_by_name(
            self.model, [name for name, _ in self.model.named_children()][-2]
        )
        if head_type == "gem":
            assert p is not None
            self.features_dim = get_module_attr_by_name_recursively(
                head, 0, "in_features"
            )
            replace_module_by_identity(self.model, head, nn.Identity())
            if gem_trainable:
                replace_module_by_identity(
                    self.model, avg_pool, GeMTrainable(p=p)
                )
            else:
                replace_module_by_identity(self.model, avg_pool, GeM(p=p))

        else:
            assert embedding_size is not None
            assert kernel_size is not None
            assert proj_hidden_dim is not None
            last_encoder_layer = get_module_by_name(
                self.model,
                [name for name, _ in self.model.named_children()][-3],
            )
            out_channels = get_module_attr_by_name_recursively(
                last_encoder_layer, -1, "out_channels"
            )
            if not out_channels:
                # Transformer based models don't have conv layers, which have
                # out_channels attr, so need to check for out_features
                out_channels = get_module_attr_by_name_recursively(
                    last_encoder_layer, -1, "out_features"
                )
            replace_module_by_identity(
                self.model,
                avg_pool,
                nn.Sequential(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        groups=out_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=(1, 1),
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.Flatten(),
                ),
            )
            replace_module_by_identity(
                self.model,
                head,
                nn.Sequential(
                    nn.Linear(proj_hidden_dim, embedding_size, bias=True),
                    nn.BatchNorm1d(embedding_size),
                ),
            )
            self.features_dim = embedding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
