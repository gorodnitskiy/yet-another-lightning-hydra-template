from typing import Any, List

import torch
from omegaconf import DictConfig

from src.modules.components.lit_module import BaseLitModule
from src.modules.losses.losses import load_loss
from src.modules.metrics.metrics import load_metrics


class MultipleLitModule(BaseLitModule):
    def __init__(
        self,
        network: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        logging: DictConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            network, optimizer, scheduler, logging, *args, **kwargs
        )
        self.loss = load_loss(network.loss)
        (
            self.total_valid_metric,
            self.total_valid_metric_best,
            _,
        ) = load_metrics(network.metrics)
        self.train_metric, _, self.train_add_metrics = load_metrics(
            network.metrics
        )
        self.valid_metric, _, self.valid_add_metrics = load_metrics(
            network.metrics
        )
        self.test_metric, _, self.test_add_metrics = load_metrics(
            network.metrics
        )
        self.parts = network.parts
        self.save_hyperparameters(logger=False)

    def on_train_start(self) -> None:
        self.total_valid_metric_best.reset()

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        loss = None
        outputs = {"preds": {}, "targets": {}}
        for idx, part in enumerate(self.parts):
            logits = self.forward(batch[part]["image"])[idx]
            preds = torch.softmax(logits, dim=1)
            targets = batch[part]["label"]
            outputs["preds"][part] = preds
            outputs["targets"][part] = targets

            curr_loss = self.loss(logits, targets)
            self.log(
                f"{self.loss.__class__.__name__}/train_{part}",
                curr_loss,
                **self.logging_params,
            )
            if idx == 0:
                loss = curr_loss
            else:
                loss += curr_loss

            self.train_metric(preds, targets)
            self.log(
                f"{self.train_metric.__class__.__name__}/train_{part}",
                self.train_metric,
                **self.logging_params,
            )

            for train_add_metric in self.train_add_metrics:
                add_metric_value = train_add_metric(preds, targets)
                self.log(
                    f"{train_add_metric.__class__.__name__}/train_{part}",
                    add_metric_value,
                    **self.logging_params,
                )

        self.log(
            f"{self.loss.__class__.__name__}/train",
            loss,
            **self.logging_params,
        )
        outputs["loss"] = loss
        return outputs

    def training_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        part = self.parts[dataloader_idx]
        logits = self.forward(batch["image"])[dataloader_idx]
        preds = torch.softmax(logits, dim=1)
        targets = batch["label"]

        loss = self.loss(logits, targets)
        self.log(
            f"{self.loss.__class__.__name__}/valid_{part}",
            loss,
            **self.logging_params,
        )

        self.valid_metric(preds, targets)
        self.log(
            f"{self.valid_metric.__class__.__name__}/valid_{part}",
            self.valid_metric,
            **self.logging_params,
        )

        for valid_add_metric in self.valid_add_metrics:
            add_metric_value = valid_add_metric(preds, targets)
            self.log(
                f"{valid_add_metric.__class__.__name__}/valid_{part}",
                add_metric_value,
                **self.logging_params,
            )

        self.total_valid_metric.update(preds, targets)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        total_valid_metric = self.total_valid_metric.compute()
        self.total_valid_metric_best(total_valid_metric)
        self.log(
            f"{self.total_valid_metric.__class__.__name__}/valid_best",
            self.total_valid_metric_best.compute(),
            **self.logging_params,
        )
        self.total_valid_metric.reset()

    def test_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        part = self.parts[dataloader_idx]
        logits = self.forward(batch["image"])[dataloader_idx]
        preds = torch.softmax(logits, dim=1)
        targets = batch["label"]

        loss = self.loss(logits, targets)
        self.log(
            f"{self.loss.__class__.__name__}/test_{part}",
            loss,
            **self.logging_params,
        )

        self.test_metric(preds, targets)
        self.log(
            f"{self.test_metric.__class__.__name__}/test_{part}",
            self.test_metric,
            **self.logging_params,
        )

        for test_add_metric in self.test_add_metrics:
            add_metric_value = test_add_metric(preds, targets)
            self.log(
                f"{test_add_metric.__class__.__name__}/test_{part}",
                add_metric_value,
                **self.logging_params,
            )

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        pass
