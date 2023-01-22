from typing import Tuple

import hydra
from omegaconf import DictConfig
from torchmetrics import Metric, MetricCollection


def load_metrics(
    metrics_cfg: DictConfig,
) -> Tuple[Metric, Metric, MetricCollection]:
    """Load main metric, `best` metric tracker, MetricCollection of additional
    metrics.

    Args:
        metrics_cfg (DictConfig): Metrics config.

    Returns:
        Tuple[Metric, Metric, ModuleList]: Main metric, `best` metric tracker,
            MetricCollection of additional metrics.
    """

    main_metric = hydra.utils.instantiate(metrics_cfg.main)
    if not metrics_cfg.get("valid_best"):
        raise RuntimeError(
            "Requires valid_best metric that would track best state of "
            "Main Metric. Usually it can be MaxMetric or MinMetric."
        )
    valid_metric_best = hydra.utils.instantiate(metrics_cfg.valid_best)

    additional_metrics = []
    if metrics_cfg.get("additional"):
        for _, metric_cfg in metrics_cfg.additional.items():
            additional_metrics.append(hydra.utils.instantiate(metric_cfg))

    return main_metric, valid_metric_best, MetricCollection(additional_metrics)
