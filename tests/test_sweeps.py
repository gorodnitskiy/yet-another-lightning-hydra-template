import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

_STARTFILE = "src/train.py"
_HYDRA_OPTIONS = ["--multirun", "--config-name=mnist_train.yaml"]
_OVERRIDES = ["logger=[]"]


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path):
    """Test running all available experiment configs with fast_dev_run=True."""
    command = [
        _STARTFILE,
        *_HYDRA_OPTIONS,
        "experiment=glob(*)",
        "hydra.sweep.dir=" + str(tmp_path),
        "++trainer.fast_dev_run=true",
    ] + _OVERRIDES
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path):
    """Test default hydra sweep."""
    command = [
        _STARTFILE,
        *_HYDRA_OPTIONS,
        "hydra.sweep.dir=" + str(tmp_path),
        "module.optimizer.lr=0.005,0.01",
        "++trainer.fast_dev_run=true",
    ] + _OVERRIDES
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep_ddp_sim(tmp_path):
    """Test default hydra sweep with ddp sim."""
    command = [
        _STARTFILE,
        *_HYDRA_OPTIONS,
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "++trainer.limit_train_batches=0.01",
        "++trainer.limit_val_batches=0.1",
        "++trainer.limit_test_batches=0.1",
        "module.optimizer.lr=0.005,0.01,0.02",
    ] + _OVERRIDES
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(tmp_path):
    """Test optuna sweep."""
    command = [
        _STARTFILE,
        *_HYDRA_OPTIONS,
        "hparams_search=mnist_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.sampler.n_startup_trials=5",
        "++trainer.fast_dev_run=true",
    ] + _OVERRIDES
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep_ddp_sim(tmp_path):
    """Test optuna sweep with ddp sim."""
    command = [
        _STARTFILE,
        *_HYDRA_OPTIONS,
        "hparams_search=mnist_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=5",
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "++trainer.limit_train_batches=0.01",
        "++trainer.limit_val_batches=0.1",
        "++trainer.limit_test_batches=0.1",
    ]
    run_sh_command(command)
