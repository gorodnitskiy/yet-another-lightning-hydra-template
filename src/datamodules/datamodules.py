from typing import Any, Dict, List, Optional, Union
from collections import OrderedDict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.components.transforms import TransformsWrapper


class SingleDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        super().__init__()
        self.cfg_datasets = datasets
        self.cfg_loaders = loaders
        self.transforms = TransformsWrapper(transforms)
        self.train_set: Optional[Dataset] = None
        self.valid_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    def _get_dataset_(self, split_name: str) -> Dataset:
        self.transforms.set_stage(split_name)
        dataset: Dataset = hydra.utils.instantiate(
            self.cfg_datasets.get(split_name), transforms=self.transforms
        )
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.
        Set variables: `self.train_set`, `self.valid_set`, `self.test_set`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split
        twice!
        """
        # load and split datasets only if not loaded already
        if not self.train_set and not self.valid_set and not self.test_set:
            self.train_set = self._get_dataset_("train")
            self.valid_set = self._get_dataset_("valid")
            self.test_set = self._get_dataset_("test")

    def get_weights(self, split_name: str) -> torch.Tensor:
        dataset: Any = self._get_dataset_(split_name)
        assert hasattr(dataset, "get_labels"), \
            "Dataset should have get_labels method"
        label_list = dataset.get_labels()
        counts = np.bincount(label_list)
        weights = torch.from_numpy(1.0 / counts).float()
        return weights

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_set, **self.cfg_loaders.get("train"))

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.valid_set, **self.cfg_loaders.get("valid"))

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_set, **self.cfg_loaders.get("test"))


class MultipleDataModule(LightningDataModule):
    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        super().__init__()
        self.cfg_datasets = datasets
        self.cfg_loaders = loaders
        self.transforms = TransformsWrapper(transforms)
        self.train_set: Optional[Dict[str, Dataset]] = None
        self.valid_set: Optional[Dict[str, Dataset]] = None
        self.test_set: Optional[Dict[str, Dataset]] = None

    def _get_dataset_(self, dataset_name: str, split_name: str) -> Dataset:
        self.transforms.set_stage(split_name)
        dataset: Dataset = hydra.utils.instantiate(
            self.cfg_datasets.get(dataset_name).get(split_name),
            transforms=self.transforms,
        )
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.
        Set variables: `self.train_set`, `self.valid_set`, `self.test_set`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split
        twice!
        """
        # load and split datasets only if not loaded already
        if not self.train_set and not self.valid_set and not self.test_set:
            self.train_set = OrderedDict()
            for dataset_name in self.cfg_datasets.keys():
                dataset = self._get_dataset_(dataset_name, "train")
                self.train_set[dataset_name] = dataset
            self.valid_set = OrderedDict()
            for dataset_name in self.cfg_datasets.keys():
                dataset = self._get_dataset_(dataset_name, "valid")
                self.valid_set[dataset_name] = dataset
            self.test_set = OrderedDict()
            for dataset_name in self.cfg_datasets.keys():
                dataset = self._get_dataset_(dataset_name, "test")
                self.test_set[dataset_name] = dataset

    def get_weights(self, dataset_name: str, split_name: str) -> torch.Tensor:
        dataset: Any = self._get_dataset_(dataset_name, split_name)
        assert hasattr(dataset, "get_labels"), \
            "Dataset should have get_labels method"
        label_list = dataset.get_labels()
        counts = np.bincount(label_list)
        weights = torch.from_numpy(1.0 / counts).float()
        return weights

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        loaders = dict()
        for dataset_name, dataset in self.train_set.items():
            loaders[dataset_name] = DataLoader(
                dataset, **self.cfg_loaders.get("train")
            )
        return loaders

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        loaders = []
        for _, dataset in self.valid_set.items():
            loaders.append(
                DataLoader(dataset, **self.cfg_loaders.get("valid"))
            )
        return loaders

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        loaders = []
        for _, dataset in self.test_set.items():
            loaders.append(
                DataLoader(dataset, **self.cfg_loaders.get("test"))
            )
        return loaders
