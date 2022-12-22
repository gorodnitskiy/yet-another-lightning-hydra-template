from typing import Tuple

import torchmetrics
from omegaconf import DictConfig
from torch.nn import ModuleList
from torchmetrics import Metric

from src.modules.metrics.components.classification import (
    AccuracyManual,
    MRRManual,
    NDCGManual,
    SentiMRRManual,
)
from src.modules.metrics.components.segmentation import IoUManual


def load_metric(name: str, **kwargs) -> Metric:
    if "torchmetrics" in name:
        name = name.split("torchmetrics/")[1]
        metric = getattr(torchmetrics, name)(**kwargs)
    # manual metrics
    elif name == "AccuracyManual":
        metric = AccuracyManual(**kwargs)
    elif name == "IoUManual":
        metric = IoUManual(**kwargs)
    elif name == "NDCGManual":
        metric = NDCGManual(**kwargs)
    elif name == "MRRManual":
        metric = MRRManual(**kwargs)
    elif name == "SentiMRRManual":
        metric = SentiMRRManual(**kwargs)
    else:
        raise NotImplementedError(f"{name} metric isn't implemented.")
    return metric


def load_metrics(config: DictConfig) -> Tuple[Metric, Metric, ModuleList]:
    main_metric = load_metric(
        config.main._target_,
        **{k: v for k, v in config.main.items() if k != "_target_"},
    )
    if not config.get("valid_best"):
        raise RuntimeError(
            "Requires valid_best metric that would track best state of "
            "Main Metric . Usually it can be MaxMetric or MinMetric."
        )
    valid_metric_best = load_metric(
        config.valid_best._target_,
        **{k: v for k, v in config.valid_best.items() if k != "_target_"},
    )

    additional_metrics = []
    if config.get("additional"):
        if not config.additional.get("names"):
            raise RuntimeError(
                "If additional metrics are used,"
                "then names params as List[metric names] should be passed."
            )
        for metric_name in config.additional.names:
            if config.additional.get(metric_name):
                params = config.additional.get(metric_name)
                metric = load_metric(metric_name, **params)
            else:
                metric = load_metric(metric_name)
            additional_metrics.append(metric)

    return main_metric, valid_metric_best, ModuleList(additional_metrics)
