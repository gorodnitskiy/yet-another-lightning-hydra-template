from typing import Optional

import torch
from pytorch_lightning.utilities import FLOAT32_EPSILON
from torchmetrics import Metric


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "correct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        diff = preds.argmax(dim=1).eq(targets)
        self.correct += diff.sum()
        self.total += targets.numel()

    def compute(self) -> torch.Tensor:
        return self.correct.float() / (self.total + FLOAT32_EPSILON)


class NDCG(Metric):
    def __init__(self, dist_sync_on_step: bool = False, k: int = 10) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("ndcg", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "count", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.k = k

    def dcg_score(
        self, y_true: torch.Tensor, y_score: torch.Tensor
    ) -> torch.Tensor:
        # if sequence smaller than k
        sequence_length = y_score.shape[1]
        if sequence_length < self.k:
            k = sequence_length
        else:
            k = self.k
        _, order = torch.topk(input=y_score, k=k, largest=True)
        y_true = torch.take(y_true, order)
        gains = torch.pow(2, y_true) - 1
        discounts = torch.log2(
            torch.arange(y_true.shape[1]).type_as(y_score) + 2.0
        )
        return torch.sum(gains / discounts)

    def ndcg_score(
        self, y_true: torch.Tensor, y_score: torch.Tensor
    ) -> torch.Tensor:
        best = self.dcg_score(y_true, y_true)
        actual = self.dcg_score(y_true, y_score)
        return actual / best

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        assert preds.shape == targets.shape
        self.ndcg += self.ndcg_score(targets, preds)
        self.count += 1.0

    def compute(self) -> torch.Tensor:
        return self.ndcg / self.count


class MRR(Metric):
    def __init__(
        self, dist_sync_on_step: bool = False, k: Optional[int] = None
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("mrr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "count", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.k = k

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        assert preds.shape == targets.shape
        if self.k is None:
            order = torch.argsort(input=preds, descending=True)
        else:
            sequence_length = preds.shape[1]
            if sequence_length < self.k:
                k = sequence_length
            else:
                k = self.k
            _, order = torch.topk(input=preds, k=k, largest=True)

        y_true = torch.take(targets, order)
        rr_score = y_true / (torch.arange(y_true.shape[1]).type_as(preds) + 1)

        self.mrr += torch.sum(rr_score) / torch.sum(y_true)
        self.count += 1.0

    def compute(self) -> torch.Tensor:
        return self.mrr / self.count


class SentiMRR(Metric):
    def __init__(
        self, dist_sync_on_step: bool = False, k: Optional[int] = None
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "senti_mrr", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "count", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.k = k

    def update(
        self, y_pred: torch.Tensor, s_c: torch.Tensor, s_mean: torch.Tensor
    ) -> None:
        assert y_pred.shape == s_c.shape
        if self.k is None:
            order = torch.argsort(input=y_pred, descending=True)
        else:
            sequence_length = y_pred.shape[0]
            if sequence_length < self.k:
                k = sequence_length
            else:
                k = self.k
            _, order = torch.topk(input=y_pred, k=k, largest=True)

        s_c = torch.take(s_c, order)
        senti_rr_score = s_c / (torch.arange(s_c.shape[0]).type_as(s_c) + 1.0)
        senti_rr_score = s_mean * torch.sum(senti_rr_score)
        senti_rr_score = torch.nn.functional.relu(senti_rr_score)

        self.senti_mrr += senti_rr_score
        self.count += 1.0

    def compute(self) -> torch.Tensor:
        return self.senti_mrr / self.count


class PrecisionAtRecall(Metric):
    def __init__(
        self, dist_sync_on_step: bool = False, recall_point: float = 0.95
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "correct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "wrong", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.recall_point = recall_point

    def update(self, distances: torch.Tensor, labels: torch.Tensor) -> None:
        labels = labels[torch.argsort(distances)]
        # Sliding threshold: get first index where recall >= recall_point.
        # This is the index where the number of elements with label==below the threshold reaches a fraction of
        # 'recall_point' of the total number of elements with label==1.
        # (np.argmax returns the first occurrence of a '1' in a bool array).
        threshold_index = torch.where(
            torch.cumsum(labels, dim=0)
            >= self.recall_point * torch.sum(labels)
        )
        threshold_index = threshold_index[0][0]
        self.correct += torch.sum(labels[threshold_index:] == 0)
        self.wrong += torch.sum(labels[:threshold_index] == 0)

    def compute(self) -> torch.Tensor:
        return self.correct.float() / (
            self.correct + self.wrong + FLOAT32_EPSILON
        )
