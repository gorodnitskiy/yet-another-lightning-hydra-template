from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import DictConfig, open_dict

from src.datamodules.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int, cfg_train: DictConfig):
    with open_dict(cfg_train):
        cfg_train.datamodule.loaders.train.batch_size = batch_size

    datamodule: MNISTDataModule = hydra.utils.instantiate(
        cfg_train.datamodule, _recursive_=False
    )
    datamodule.prepare_data()

    assert not datamodule.train_set
    assert not datamodule.valid_set
    assert not datamodule.test_set
    assert not datamodule.predict_set
    assert Path(cfg_train.paths.data_dir, "MNIST").exists()
    assert Path(cfg_train.paths.data_dir, "MNIST", "raw").exists()

    datamodule.setup()
    assert datamodule.train_set
    assert datamodule.valid_set
    assert datamodule.test_set

    datamodule.setup(stage="predict")
    assert datamodule.predict_set

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()
    assert datamodule.predict_dataloader()

    train_samples = len(datamodule.train_set)
    valid_samples = len(datamodule.valid_set)
    test_samples = len(datamodule.test_set)
    assert (train_samples + valid_samples + test_samples) == 70_000

    predict_samples = [len(value) for value in datamodule.predict_set.values()]
    assert test_samples == sum(predict_samples)

    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch
    assert len(x["image"]) == batch_size
    assert len(y) == batch_size
    assert x["image"].dtype == torch.float32
    assert y.dtype == torch.int64
