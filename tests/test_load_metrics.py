from typing import Any, Dict

import omegaconf
import pytest

from src.modules.metrics import load_metrics

_TEST_METRICS_CFG = (
    {"_target_": "torchmetrics.Accuracy", "task": "binary"},
    {"_target_": "torchmetrics.AUROC", "task": "binary"},
    {"_target_": "torchmetrics.FBetaScore", "task": "binary"},
    {"_target_": "torchmetrics.AveragePrecision", "task": "binary"},
    {"_target_": "torchmetrics.JaccardIndex", "task": "binary"},
    {"_target_": "src.modules.metrics.Accuracy"},
    {"_target_": "src.modules.metrics.IoU", "n_class": 2},
    {"_target_": "src.modules.metrics.NDCG"},
    {"_target_": "src.modules.metrics.MRR"},
    {"_target_": "src.modules.metrics.SentiMRR"},
)

_TEST_METRICS_CFG_COMPLEX = (
    {
        "main": {
            "_target_": "torchmetrics.Accuracy",
            "task": "multiclass",
            "num_classes": 10,
            "top_k": 1,
        },
        "valid_best": {"_target_": "torchmetrics.MaxMetric"},
        "additional": {
            "AUROC": {
                "_target_": "torchmetrics.AUROC",
                "task": "multiclass",
                "num_classes": 10,
            },
        },
    },
    {
        "main": {"_target_": "torchmetrics.Accuracy", "task": "binary"},
        "valid_best": {"_target_": "torchmetrics.MaxMetric"},
        "additional": {
            "JaccardIndex": {
                "_target_": "torchmetrics.JaccardIndex",
                "task": "binary",
            },
            "AUROC": {"_target_": "torchmetrics.AUROC", "task": "binary"},
        },
    },
    {
        "main": {"_target_": "src.modules.metrics.Accuracy"},
        "valid_best": {"_target_": "torchmetrics.MaxMetric"},
        "additional": {
            "IoU": {
                "_target_": "src.modules.metrics.IoU",
                "n_class": 2,
            },
        },
    },
)


@pytest.mark.parametrize("metric_cfg", _TEST_METRICS_CFG)
def test_metric_cfg(metric_cfg: Dict[str, Any]):
    metrics_cfg = {
        "main": metric_cfg,
        "valid_best": {"_target_": "torchmetrics.MaxMetric"},
    }
    cfg = omegaconf.OmegaConf.create(metrics_cfg)
    _ = load_metrics(cfg)


@pytest.mark.parametrize("metrics_cfg", _TEST_METRICS_CFG_COMPLEX)
def test_metrics_cfg(metrics_cfg: Dict[str, Any]):
    cfg = omegaconf.OmegaConf.create(metrics_cfg)
    _ = load_metrics(cfg)
