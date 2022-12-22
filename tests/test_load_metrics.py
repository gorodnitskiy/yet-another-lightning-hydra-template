from typing import Any, Dict, Tuple

import omegaconf
import pytest

from src.modules.metrics.metrics import load_metric, load_metrics

_IMPLEMENTED_METRICS = (
    ("torchmetrics/Accuracy", {"task": "binary"}),
    ("torchmetrics/AUROC", {"task": "binary"}),
    ("torchmetrics/FBetaScore", {"task": "binary"}),
    ("torchmetrics/AveragePrecision", {"task": "binary"}),
    ("torchmetrics/JaccardIndex", {"task": "binary"}),
    ("AccuracyManual", {}),
    ("IoUManual", {"n_class": 2}),
    ("NDCGManual", {}),
    ("MRRManual", {}),
    ("SentiMRRManual", {}),
)

_CFG_METRICS = [
    {
        "main": {
            "_target_": "torchmetrics/Accuracy",
            "task": "multiclass",
            "num_classes": 10,
            "top_k": 1,
        },
        "valid_best": {"_target_": "torchmetrics/MaxMetric"},
        "additional": {
            "names": ["torchmetrics/AUROC"],
            "torchmetrics/AUROC": {"task": "multiclass", "num_classes": 10},
        },
    },
    {
        "main": {"_target_": "torchmetrics/Accuracy", "task": "binary"},
        "valid_best": {"_target_": "torchmetrics/MaxMetric"},
        "additional": {
            "names": ["torchmetrics/JaccardIndex", "torchmetrics/AUROC"],
            "torchmetrics/JaccardIndex": {"task": "binary"},
            "torchmetrics/AUROC": {"task": "binary"},
        },
    },
    {
        "main": {"_target_": "AccuracyManual"},
        "valid_best": {"_target_": "torchmetrics/MaxMetric"},
    },
]


@pytest.mark.parametrize("metric", _IMPLEMENTED_METRICS)
def test_metric(metric: Tuple[str, Dict[str, Any]]):
    name, params = metric
    _ = load_metric(name, **params)


@pytest.mark.parametrize("cfg_metric", _CFG_METRICS)
def test_metrics(cfg_metric: Dict[str, Any]):
    cfg = omegaconf.OmegaConf.create(cfg_metric)
    _ = load_metrics(cfg)
