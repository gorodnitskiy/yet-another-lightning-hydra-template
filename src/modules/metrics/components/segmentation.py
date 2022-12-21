import torch
from torchmetrics import Metric


class IoUManual(Metric):
    def __init__(
        self,
        n_class: int,
        dist_sync_on_step: bool = False,
        dist_reduce_fx: str = "sum",
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.n_class = n_class
        self.add_state(
            "intersection",
            default=torch.zeros((n_class,)),
            dist_reduce_fx=dist_reduce_fx,
        )
        self.add_state(
            "union",
            default=torch.zeros((n_class,)),
            dist_reduce_fx=dist_reduce_fx,
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        res = preds.argmax(dim=1)
        for index in range(self.n_class):
            gt = target.cpu() == index
            preds = res == index

            intersection = gt.logical_and(preds.cpu())
            union = gt.logical_or(preds.cpu())

            self.intersection[index] += intersection.float().sum()
            self.union[index] += union.float().sum()

    def compute(self) -> torch.Tensor:
        return self.intersection.sum() / self.union.sum()
