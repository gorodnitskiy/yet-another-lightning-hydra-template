from collections import OrderedDict
from typing import Dict, List, Optional, Union

import hydra
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
        def predict_dataloader(self):
            # return predict dataloader
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
        self.predict_set: Dict[str, Dataset] = OrderedDict()

    def _get_dataset_(
        self, split_name: str, dataset_name: Optional[str] = None
    ) -> Dataset:
        self.transforms.set_stage(split_name)
        cfg = self.cfg_datasets.get(split_name)
        if dataset_name:
            cfg = cfg.get(dataset_name)
        dataset: Dataset = hydra.utils.instantiate(
            cfg, transforms=self.transforms
        )
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.train_set`, `self.valid_set`,
        `self.test_set`, `self.predict_set`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split
        twice!
        """
        # load and split datasets only if not loaded already
        if not self.train_set and not self.valid_set and not self.test_set:
            self.train_set = self._get_dataset_("train")
            self.valid_set = self._get_dataset_("valid")
            self.test_set = self._get_dataset_("test")
        # load predict datasets only if it exists in config
        if (stage == "predict") and self.cfg_datasets.get("predict"):
            for dataset_name in self.cfg_datasets.get("predict").keys():
                self.predict_set[dataset_name] = self._get_dataset_(
                    "predict", dataset_name=dataset_name
                )

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_set, **self.cfg_loaders.get("train"))

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.valid_set, **self.cfg_loaders.get("valid"))

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_set, **self.cfg_loaders.get("test"))

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        loaders = []
        for _, dataset in self.predict_set.items():
            loaders.append(
                DataLoader(dataset, **self.cfg_loaders.get("predict"))
            )
        return loaders

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass


class MultipleDataModule(SingleDataModule):
    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        super().__init__(
            datasets=datasets, loaders=loaders, transforms=transforms
        )
        self.train_set: Optional[Dict[str, Dataset]] = None
        self.valid_set: Optional[Dict[str, Dataset]] = None
        self.test_set: Optional[Dict[str, Dataset]] = None
        self.predict_set: Dict[str, Dataset] = OrderedDict()

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.train_set`, `self.valid_set`,
        `self.test_set`, `self.predict_set`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split
        twice!
        """
        # load and split datasets only if not loaded already
        if not self.train_set and not self.valid_set and not self.test_set:
            self.train_set = OrderedDict()
            for dataset_name in self.cfg_datasets.get("train").keys():
                self.train_set[dataset_name] = self._get_dataset_(
                    "train", dataset_name=dataset_name
                )
            self.valid_set = OrderedDict()
            for dataset_name in self.cfg_datasets.get("valid").keys():
                self.valid_set[dataset_name] = self._get_dataset_(
                    "valid", dataset_name=dataset_name
                )
            self.test_set = OrderedDict()
            for dataset_name in self.cfg_datasets.get("test").keys():
                self.test_set[dataset_name] = self._get_dataset_(
                    "test", dataset_name=dataset_name
                )
        # load predict datasets only if it exists in config
        if (stage == "predict") and self.cfg_datasets.get("predict"):
            for dataset_name in self.cfg_datasets.get("predict").keys():
                self.predict_set[dataset_name] = self._get_dataset_(
                    "predict", dataset_name=dataset_name
                )

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
            loaders.append(DataLoader(dataset, **self.cfg_loaders.get("test")))
        return loaders

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        loaders = []
        for _, dataset in self.predict_set.items():
            loaders.append(
                DataLoader(dataset, **self.cfg_loaders.get("predict"))
            )
        return loaders
