import warnings
from functools import wraps
from importlib.util import find_spec
from typing import Any, Callable, List, Optional

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from src.modules.losses import load_loss
from src.modules.metrics import load_metrics
from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished or failed
    - Logging the exception if occurs
    - Logging the output dir

    Args:
        task_func (Callable): Task function.

    Returns:
        Callable: Decorator that wraps the task function in extra utilities.
    """

    def wrap(cfg: DictConfig):

        # execute the task
        try:

            # apply extra utilities
            extras(cfg)

            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:

            # save exception to `.log` file
            log.exception("")

            # when using hydra plugins like Optuna, you might want to disable
            # raising exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:

            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # close loggers (even if exception occurs so multirun won't fail)
            close_loggers()

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing

    Args:
        cfg (DictConfig): Main config.
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info(
            "Disabling python warnings! <cfg.extras.ignore_warnings=True>"
        )
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info(
            "Printing config tree with Rich! <cfg.extras.print_config=True>"
        )
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    Args:
        callbacks_cfg (DictConfig): Callbacks config.

    Returns:
        List[Callback]: List with all instantiated callbacks.
    """

    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[LightningLoggerBase]:
    """Instantiates loggers from config.

    Args:
        logger_cfg (DictConfig): Loggers config.

    Returns:
        List[LightningLoggerBase]: List with all instantiated loggers.
    """

    logger: List[LightningLoggerBase] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Saves additionally:
    - Number of model parameters

    Args:
        object_dict (dict): Dict object with all parameters.
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["module"] = cfg["module"]

    # save number of model parameters
    hparams["module/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["module/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["module/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    Args:
        metric_dict (dict): Dict with metric values.
        metric_name (str): Metric name.

    Returns:
        Optional[float]: Metric value.
    """

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during
    multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup).

    Args:
        path (str): File path.
        content (str): File content.
    """

    with open(path, "w+") as file:
        file.write(content)


def instantiate_plugins(cfg: DictConfig) -> Optional[List[Any]]:
    """Instantiates lightning plugins from config.

    Args:
        cfg (DictConfig): Config.

    Returns:
        List[Any]: List with all instantiated plugins.
    """

    if not cfg.extras.get("plugins"):
        log.warning("No plugins configs found! Skipping...")
        return

    if cfg.trainer.get("accelerator") == "cpu":
        log.warning("Using CPU as accelerator! Skipping...")
        return

    plugins: List[Any] = []
    for _, pl_conf in cfg.extras.get("plugins").items():
        if isinstance(pl_conf, DictConfig) and "_target_" in pl_conf:
            log.info(f"Instantiating plugin <{pl_conf._target_}>")
            plugins.append(hydra.utils.instantiate(pl_conf))

    return plugins


def register_custom_resolvers(
    version_base: str, config_path: str, config_name: str
) -> Callable:
    """Optional decorator to register custom OmegaConf resolvers. It is
    excepted to call before `hydra.main` decorator call.

    Replace resolver: To avoiding copying of loss and metric names in configs,
    there is custom resolver during hydra initialization which replaces
    `__loss__` to `loss.__class__.__name__` and `__metric__` to
    `main_metric.__class__.__name__` For example: ${replace:"__metric__/valid"}
    Use quotes for defining internal value in ${replace:"..."} to avoid grammar
    problems with hydra config parser.

    Args:
        version_base (str): Hydra version base.
        config_path (str): Hydra config path.
        config_name (str): Hydra config name.

    Returns:
        Callable: Decorator that registers custom resolvers before running
            main function.
    """

    # register of replace resolver
    if not OmegaConf.has_resolver("replace"):
        with initialize_config_dir(
            version_base=version_base, config_dir=config_path
        ):
            cfg = compose(
                config_name=config_name, return_hydra_config=True, overrides=[]
            )
        cfg_tmp = cfg.copy()
        loss = load_loss(cfg_tmp.module.network.loss)
        metric, _, _ = load_metrics(cfg_tmp.module.network.metrics)
        GlobalHydra.instance().clear()

        OmegaConf.register_new_resolver(
            "replace",
            lambda item: item.replace(
                "__loss__", loss.__class__.__name__
            ).replace("__metric__", metric.__class__.__name__),
        )

    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return function(*args, **kwargs)

        return wrapper

    return decorator
