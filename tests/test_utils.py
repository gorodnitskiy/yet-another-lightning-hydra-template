import os

from omegaconf import DictConfig

from src import utils


def test_rich_utils(cfg_train: DictConfig):
    utils.print_config_tree(cfg_train, resolve=False, save_to_file=False)


def test_metadata_utils(tmp_path, cfg_train):
    utils.log_metadata(cfg_train)

    files = os.listdir(tmp_path / "metadata")
    assert "pip.log" in files
    assert "git.log" in files
    assert "gpu.log" in files
    assert "src" in files
    assert "configs" in files
