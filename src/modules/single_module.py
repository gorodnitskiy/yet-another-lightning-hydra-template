from typing import Any, List

import hydra
import torch
from omegaconf import DictConfig
from torch import nn

from src.modules.components.lit_module import BaseLitModule
from src.modules.losses import load_loss
from src.modules.metrics import load_metrics


class SingleLitModule(BaseLitModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Model loop (model_step)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        network: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        logging: DictConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """LightningModule with standalone train, val and test dataloaders.

        Args:
            network (DictConfig): Network config.
            optimizer (DictConfig): Optimizer config.
            scheduler (DictConfig): Scheduler config.
            logging (DictConfig): Logging config.
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
        self.train_metric, _, self.train_add_metrics = load_metrics(
            network.metrics
        )
        (
            self.valid_metric,
            self.valid_metric_best,
            self.valid_add_metrics,
        ) = load_metrics(network.metrics)
        self.test_metric, _, self.test_add_metrics = load_metrics(
            network.metrics
        )
        self.save_hyperparameters(logger=False)

    def model_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        logits = self.forward(batch["image"])
        loss = self.loss(logits, batch["label"])
        preds = self.output_activation(logits)
        return loss, preds, batch["label"]

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before
        # training starts, so we need to make sure valid_metric_best doesn't store
        # accuracy from these checks
        self.valid_metric_best.reset()

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f"{self.loss.__class__.__name__}/train",
            loss,
            **self.logging_params,
        )

        self.train_metric(preds, targets)
        self.log(
            f"{self.train_metric.__class__.__name__}/train",
            self.train_metric,
            **self.logging_params,
        )

        for train_add_metric in self.train_add_metrics:
            add_metric_value = train_add_metric(preds, targets)
            self.log(
                f"{train_add_metric.__class__.__name__}/train",
                add_metric_value,
                **self.logging_params,
            )
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]) -> None:
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning
        # accumulates outputs from all batches of the epoch

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f"{self.loss.__class__.__name__}/valid",
            loss,
            **self.logging_params,
        )

        self.valid_metric(preds, targets)
        self.log(
            f"{self.valid_metric.__class__.__name__}/valid",
            self.valid_metric,
            **self.logging_params,
        )

        for valid_add_metric in self.valid_add_metrics:
            add_metric_value = valid_add_metric(preds, targets)
            self.log(
                f"{valid_add_metric.__class__.__name__}/valid",
                add_metric_value,
                **self.logging_params,
            )
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        valid_metric = self.valid_metric.compute()  # get current valid metric
        self.valid_metric_best(valid_metric)  # update best so far valid metric
        # log `valid_metric_best` as a value through `.compute()` method, instead
        # of as a metric object otherwise metric would be reset by lightning
        # after each epoch
        self.log(
            f"{self.valid_metric.__class__.__name__}/valid_best",
            self.valid_metric_best.compute(),
            **self.logging_params,
        )

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f"{self.loss.__class__.__name__}/test", loss, **self.logging_params
        )

        self.test_metric(preds, targets)
        self.log(
            f"{self.test_metric.__class__.__name__}/test",
            self.test_metric,
            **self.logging_params,
        )

        for test_add_metric in self.test_add_metrics:
            add_metric_value = test_add_metric(preds, targets)
            self.log(
                f"{test_add_metric.__class__.__name__}/test",
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
        preds = self.output_activation(logits)
        outputs = {"logits": logits, "preds": preds}
        if "label" in batch:
            outputs.update({"targets": batch["label"]})
        if "name" in batch:
            outputs.update({"names": batch["name"]})
        return outputs


class MNISTLitModule(SingleLitModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def model_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        x, y = batch
        logits = self.forward(x["image"])
        loss = self.loss(logits, y)
        preds = self.output_activation(logits)
        return loss, preds, y

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        x, y = batch
        logits = self.forward(x["image"])
        preds = self.output_activation(logits)
        return {"logits": logits, "preds": preds, "targets": y}


class SingleVicRegLitModule(BaseLitModule):
    def __init__(
        self,
        network: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        logging: DictConfig,
        proj_hidden_dim: int,
        proj_output_dim: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """LightningModule with standalone train, val and test dataloaders for
        Self-Supervised task using VicReg approach.

        Args:
            network (DictConfig): Network config.
            optimizer (DictConfig): Optimizer config.
            scheduler (DictConfig): Scheduler config.
            logging (DictConfig): Logging config.
            proj_hidden_dim (int): Projector hidden dimensions.
            proj_output_dim (int): Projector output dimensions.
            args (Any): Additional arguments for pytorch_lightning.LightningModule.
            kwargs (Any): Additional keyword arguments for pytorch_lightning.LightningModule.
        """

        super().__init__(
            network, optimizer, scheduler, logging, *args, **kwargs
        )
        self.loss = load_loss(network.loss)
        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.model.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            # nn.Linear(proj_hidden_dim, proj_hidden_dim),
            # nn.BatchNorm1d(proj_hidden_dim),
            # nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        self.save_hyperparameters(logger=False)

    def forward(self, x: Any) -> Any:
        x = self.model.forward(x)
        return self.projector(x)

    def model_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        z1 = self.forward(batch["z1"])
        z2 = self.forward(batch["z2"])
        loss = self.loss(z1, z2)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        loss = self.model_step(batch, batch_idx)
        self.log(
            f"{self.loss.__class__.__name__}/train",
            loss,
            **self.logging_params,
        )
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        loss = self.model_step(batch, batch_idx)
        self.log(
            f"{self.loss.__class__.__name__}/valid",
            loss,
            **self.logging_params,
        )
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        loss = self.model_step(batch, batch_idx)
        self.log(
            f"{self.loss.__class__.__name__}/test", loss, **self.logging_params
        )
        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        pass


class SingleReIdLitModule(SingleLitModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def model_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        embeddings = self.forward(batch["image"])
        return embeddings, batch["label"]

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        embeddings, targets = self.model_step(batch, batch_idx)
        loss, logits = self.loss(embeddings, batch["label"])
        preds = self.output_activation(logits)
        self.log(
            f"{self.loss.__class__.__name__}/train",
            loss,
            **self.logging_params,
        )

        self.train_metric(preds, targets)
        self.log(
            f"{self.train_metric.__class__.__name__}/train",
            self.train_metric,
            **self.logging_params,
        )
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        embeddings, targets = self.model_step(batch, batch_idx)
        with torch.no_grad():
            loss, logits = self.loss(embeddings, batch["label"])
        preds = self.output_activation(logits)
        self.log(
            f"{self.loss.__class__.__name__}/valid",
            loss,
            **self.logging_params,
        )

        self.valid_metric(preds, targets)
        self.log(
            f"{self.valid_metric.__class__.__name__}/valid",
            self.valid_metric,
            **self.logging_params,
        )
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        embeddings, targets = self.model_step(batch, batch_idx)
        with torch.no_grad():
            loss, logits = self.loss(embeddings, batch["label"])
        preds = self.output_activation(logits)
        self.log(
            f"{self.loss.__class__.__name__}/test", loss, **self.logging_params
        )

        self.test_metric(preds, targets)
        self.log(
            f"{self.test_metric.__class__.__name__}/test",
            self.test_metric,
            **self.logging_params,
        )
        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        outputs = {"embeddings": self.forward(batch["image"])}
        if "name" in batch:
            outputs.update({"names": batch["name"]})
        return outputs
