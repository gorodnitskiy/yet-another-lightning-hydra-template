from typing import Any, List

import hydra
import torch
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
        self.heads = heads

        main_metric, valid_metric_best, add_metrics = load_metrics(
            network.metrics
        )
        for head in heads:
            for step in ("train", "valid", "test"):
                setattr(self, f"{step}_metric_{head}", main_metric.clone())
                setattr(
                    self,
                    f"{step}_add_metrics_{head}",
                    add_metrics.clone(postfix=f"/{step}_{head}"),
                )
        self.total_valid_metric = main_metric.clone()
        self.total_valid_metric_best = valid_metric_best.clone()

        self.save_hyperparameters(logger=False)

    def on_train_start(self) -> None:
        self.total_valid_metric_best.reset()

    def log_metrics(
        self, step: str, head: str, preds: torch.Tensor, targets: torch.Tensor
    ) -> None:
        metric = getattr(self, f"{step}_metric_{head}")
        metric(preds, targets)
        self.log(
            f"{metric.__class__.__name__}/{step}_{head}",
            metric,
            **self.logging_params,
        )

        add_metrics = getattr(self, f"{step}_add_metrics_{head}")
        add_metrics(preds, targets)
        self.log_dict(add_metrics, **self.logging_params)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        losses = []
        for idx, head in enumerate(self.heads):
            logits = self.forward(batch[head]["image"])[idx]
            preds = self.output_activation(logits)
            targets = batch[head]["label"]

            loss = self.loss(logits, targets)
            self.log(
                f"{self.loss.__class__.__name__}/train_{head}",
                loss,
                **self.logging_params,
            )
            losses.append(loss)

            self.log_metrics("train", head, preds, targets)

        loss = sum(losses)
        self.log(
            f"{self.loss.__class__.__name__}/train",
            loss,
            **self.logging_params,
        )
        return {"loss": loss}

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

        self.log_metrics("valid", head, preds, targets)
        self.total_valid_metric.update(preds, targets)
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        total_valid_metric = self.total_valid_metric.compute()
        self.total_valid_metric_best(total_valid_metric)
        self.log(
            f"{self.total_valid_metric.__class__.__name__}/valid_best",
            self.total_valid_metric_best.compute(),
            **self.logging_params,
        )

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

        self.log_metrics("test", head, preds, targets)
        return {"loss": loss}

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
