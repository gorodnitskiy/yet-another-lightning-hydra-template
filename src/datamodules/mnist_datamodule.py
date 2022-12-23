from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, random_split
from torchvision.datasets import MNIST

from src.datamodules.datamodules import SingleDataModule


class MNISTDataModule(SingleDataModule):
    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        super().__init__(
            datasets=datasets, loaders=loaders, transforms=transforms
        )

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        MNIST(self.cfg_datasets.get("data_dir"), train=True, download=True)
        MNIST(self.cfg_datasets.get("data_dir"), train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # load and split datasets only if not loaded already
        if not self.train_set and not self.valid_set and not self.test_set:
            self.transforms.set_stage("train")
            train_set = MNIST(
                self.cfg_datasets.get("data_dir"),
                train=True,
                transform=self.transforms,
            )
            self.transforms.set_stage("test")
            test_set = MNIST(
                self.cfg_datasets.get("data_dir"),
                train=False,
                transform=self.transforms,
            )
            dataset = ConcatDataset(datasets=[train_set, test_set])
            seed = self.cfg_datasets.get("seed")
            self.train_set, self.valid_set, self.test_set = random_split(
                dataset=dataset,
                lengths=self.cfg_datasets.get("train_val_test_split"),
                generator=torch.Generator().manual_seed(seed),
            )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
