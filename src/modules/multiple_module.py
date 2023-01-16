from typing import Any, List

import hydra
from omegaconf import DictConfig

from src.modules.components.lit_module import BaseLitModule
from src.modules.losses import load_loss
from src.modules.metrics import load_metrics


class MultipleLitModule(BaseLitModule):
    def __init__(
        self,
        network: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        logging: DictConfig,
        heads: DictConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """LightningModule with multiple train, val and test dataloaders.

        Args:
            network (DictConfig): Network config.
            optimizer (DictConfig): Optimizer config.
            scheduler (DictConfig): Scheduler config.
            logging (DictConfig): Logging config.
            heads: (DictConfig): List of output heads names.
            args (Any): Additional arguments for pytorch_lightning.LightningModule.
            kwargs (Any): Additional keyword arguments for pytorch_lightning.LightningModule.
        """

        super().__init__(
            network, optimizer, scheduler, logging, *args, **kwargs
        )
        self.loss = load_loss(network.loss)
        self.output_activation = hydra.utils.instantiate(
            network.output_activation, _partial_=True
        )
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
        self.heads = heads
        self.save_hyperparameters(logger=False)

    def on_train_start(self) -> None:
        self.total_valid_metric_best.reset()

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        loss = None
        outputs = {"preds": {}, "targets": {}}
        for idx, head in enumerate(self.heads):
            logits = self.forward(batch[head]["image"])[idx]
            preds = self.output_activation(logits)
            targets = batch[head]["label"]
            outputs["preds"][head] = preds
            outputs["targets"][head] = targets

            curr_loss = self.loss(logits, targets)
            self.log(
                f"{self.loss.__class__.__name__}/train_{head}",
                curr_loss,
                **self.logging_params,
            )
            if idx == 0:
                loss = curr_loss
            else:
                loss += curr_loss

            self.train_metric(preds, targets)
            self.log(
                f"{self.train_metric.__class__.__name__}/train_{head}",
                self.train_metric,
                **self.logging_params,
            )

            for train_add_metric in self.train_add_metrics:
                add_metric_value = train_add_metric(preds, targets)
                self.log(
                    f"{train_add_metric.__class__.__name__}/train_{head}",
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
        head = self.heads[dataloader_idx]
        logits = self.forward(batch["image"])[dataloader_idx]
        preds = self.output_activation(logits)
        targets = batch["label"]

        loss = self.loss(logits, targets)
        self.log(
            f"{self.loss.__class__.__name__}/valid_{head}",
            loss,
            **self.logging_params,
        )

        self.valid_metric(preds, targets)
        self.log(
            f"{self.valid_metric.__class__.__name__}/valid_{head}",
            self.valid_metric,
            **self.logging_params,
        )

        for valid_add_metric in self.valid_add_metrics:
            add_metric_value = valid_add_metric(preds, targets)
            self.log(
                f"{valid_add_metric.__class__.__name__}/valid_{head}",
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
        head = self.heads[dataloader_idx]
        logits = self.forward(batch["image"])[dataloader_idx]
        preds = self.output_activation(logits)
        targets = batch["label"]

        loss = self.loss(logits, targets)
        self.log(
            f"{self.loss.__class__.__name__}/test_{head}",
            loss,
            **self.logging_params,
        )

        self.test_metric(preds, targets)
        self.log(
            f"{self.test_metric.__class__.__name__}/test_{head}",
            self.test_metric,
            **self.logging_params,
        )

        for test_add_metric in self.test_add_metrics:
            add_metric_value = test_add_metric(preds, targets)
            self.log(
                f"{test_add_metric.__class__.__name__}/test_{head}",
                add_metric_value,
                **self.logging_params,
            )

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        logits = self.forward(batch["image"])
        outputs = {"logits": {}, "preds": {}}
        for idx, head in enumerate(self.heads):
            outputs["logits"][head] = logits[idx]
            outputs["preds"][head] = self.output_activation(logits[idx])
        if "label" in batch:
            outputs.update({"targets": batch["label"]})
        if "name" in batch:
            outputs.update({"names": batch["name"]})
        return outputs
