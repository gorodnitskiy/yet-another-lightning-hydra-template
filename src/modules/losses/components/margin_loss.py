from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class AngularPenaltySMLoss(torch.nn.Module):
    available_losses = ("arcface", "sphereface", "cosface")

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        loss_type: str = "CosFace",
        scale: Optional[float] = None,
        margin: Optional[float] = None,
        eps: float = 1e-7,
    ) -> None:
        """
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']

        These losses are described in the following papers:
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        assert self.loss_type in self.available_losses
        if self.loss_type == "arcface":
            self.scale = 64.0 if not scale else scale
            self.margin = 0.5 if not margin else margin
        elif self.loss_type == "sphereface":
            self.scale = 64.0 if not scale else scale
            self.margin = 1.35 if not margin else margin
        elif self.loss_type == "cosface":
            self.scale = 30.0 if not scale else scale
            self.margin = 0.4 if not margin else margin
        else:
            raise NotImplementedError(f"Use one of {self.available_losses}")

        self.scale = torch.tensor(self.scale, dtype=torch.float32)
        self.margin = torch.tensor(self.margin, dtype=torch.float32)
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.weight = torch.nn.Parameter(
            torch.Tensor(num_classes, embedding_size)
        )
        torch.nn.init.xavier_uniform_(self.weight)
        self.eps = eps

    def forward(
        self, input: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        input shape: (B, embedding_size)
        """
        assert len(input) == len(label)
        assert torch.min(label) >= 0
        assert torch.max(label) < self.num_classes
        cosine = F.linear(
            F.normalize(input, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        )
        if self.loss_type == "cosface":
            numerator = torch.diagonal(cosine.transpose(0, 1)[label])
            numerator = self.scale * (numerator - self.margin)
        elif self.loss_type == "arcface":
            numerator = torch.diagonal(cosine.transpose(0, 1)[label])
            numerator = torch.clamp(numerator, -1.0 + self.eps, 1 - self.eps)
            numerator = torch.acos(numerator)
            numerator = self.scale * torch.cos(numerator + self.margin)
        elif self.loss_type == "sphereface":
            numerator = torch.diagonal(cosine.transpose(0, 1)[label])
            numerator = torch.clamp(numerator, -1.0 + self.eps, 1 - self.eps)
            numerator = torch.acos(numerator)
            numerator = self.scale * torch.cos(self.margin * numerator)
        else:
            raise NotImplementedError(f"Use one of {self.available_losses}")

        excl = []
        for i, y in enumerate(label):
            item = torch.cat((cosine[i, :y], cosine[i, y + 1 :])).unsqueeze(0)
            excl.append(item)
        excl = torch.exp(self.scale * torch.cat(excl, dim=0))
        denominator = torch.exp(numerator) + torch.sum(excl, dim=1)
        loss = numerator - torch.log(denominator)
        return -torch.mean(loss), cosine
