from omegaconf import DictConfig

from src.utils import print_config_tree


def test_rich_utils(cfg_train: DictConfig):
    print_config_tree(cfg_train, resolve=False, save_to_file=False)
