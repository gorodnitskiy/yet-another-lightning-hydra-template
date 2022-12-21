import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_train_config(cfg_train: DictConfig):
    assert cfg_train
    assert cfg_train.datamodule
    assert cfg_train.module
    assert cfg_train.module.network.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.datamodule, _recursive_=False)
    hydra.utils.instantiate(cfg_train.module, _recursive_=False)
    hydra.utils.instantiate(cfg_train.module.network.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig):
    assert cfg_eval
    assert cfg_eval.datamodule
    assert cfg_eval.module
    assert cfg_eval.module.network.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.datamodule, _recursive_=False)
    hydra.utils.instantiate(cfg_eval.module, _recursive_=False)
    hydra.utils.instantiate(cfg_eval.module.network.model)
    hydra.utils.instantiate(cfg_eval.trainer)
