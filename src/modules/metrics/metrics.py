from typing import Tuple

import hydra
from omegaconf import DictConfig
from torch.nn import ModuleList
from torchmetrics import Metric


def load_metrics(metrics_cfg: DictConfig) -> Tuple[Metric, Metric, ModuleList]:
    """Load main metric, `best` metric tracker, ModuleList of additional
    metrics.

    Args:
        metrics_cfg (DictConfig): Metrics config.

    Returns:
        Tuple[Metric, Metric, ModuleList]: Main metric, `best` metric tracker,
            ModuleList of additional metrics.
    """

    main_metric = hydra.utils.instantiate(metrics_cfg.main)
    if not metrics_cfg.get("valid_best"):
        raise RuntimeError(
            "Requires valid_best metric that would track best state of "
            "Main Metric . Usually it can be MaxMetric or MinMetric."
        )
    valid_metric_best = hydra.utils.instantiate(metrics_cfg.valid_best)

    additional_metrics = []
    if metrics_cfg.get("additional"):
        for _, metric_cfg in metrics_cfg.additional.items():
            additional_metrics.append(hydra.utils.instantiate(metric_cfg))

    return main_metric, valid_metric_best, ModuleList(additional_metrics)
