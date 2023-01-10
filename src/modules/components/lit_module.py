from typing import Any, Callable, List, Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader


class BaseLitModule(LightningModule):
    def __init__(
        self,
        network: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        logging: DictConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """BaseLightningModule.

        Args:
            network (DictConfig): Network config.
            optimizer (DictConfig): Optimizer config.
            scheduler (DictConfig): Scheduler config.
            logging (DictConfig): Logging config.
            args (Any): Additional arguments for pytorch_lightning.LightningModule.
            kwargs (Any): Additional keyword arguments for pytorch_lightning.LightningModule.
        """

        super().__init__(*args, **kwargs)
        self.model = hydra.utils.instantiate(network.model)
        self.opt_params = optimizer
        self.slr_params = scheduler
        self.logging_params = logging

    def forward(self, x: Any) -> Any:
        return self.model.forward(x)

    def configure_optimizers(self) -> Any:
        optimizer: torch.optim = hydra.utils.instantiate(
            self.opt_params, params=self.parameters(), _convert_="partial"
        )
        if self.slr_params.get("scheduler"):
            scheduler: torch.optim.lr_scheduler = hydra.utils.instantiate(
                self.slr_params.scheduler,
                optimizer=optimizer,
                _convert_="partial",
            )
            lr_scheduler_dict = {"scheduler": scheduler}
            if self.slr_params.get("extras"):
                for key, value in self.slr_params.get("extras").items():
                    lr_scheduler_dict[key] = value
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        return {"optimizer": optimizer}

    def check_grad_cam(
        self,
        dataloader: DataLoader,
        target_layer: List[Any],
        target_category: Any,
        dataloader_idx: Optional[int] = None,
        reshape_transform: Optional[Callable] = None,
        use_cuda: bool = False,
    ) -> Tuple[Any, ...]:
        grad_cam = GradCAMPlusPlus(
            model=self,
            target_layer=target_layer,
            use_cuda=use_cuda,
            reshape_transform=reshape_transform,
        )

        batch = next(iter(dataloader))
        images, labels = batch["image"], batch["label"]
        grad_cam.batch_size = len(images)
        logits = self.forward(images)
        if dataloader_idx:
            logits = logits[dataloader_idx]
        logits = logits.squeeze(1)
        pred = logits.sigmoid().detach().cpu().numpy() * 100
        labels = labels.cpu().numpy()

        grayscale_cam = grad_cam(
            input_tensor=images,
            target_category=target_category,
            eigen_smooth=True,
        )
        images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
        images -= images.min()
        images /= images.max()
        return images, grayscale_cam, pred, labels
