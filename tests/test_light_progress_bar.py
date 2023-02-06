from typing import Any, Dict

from src.callbacks.light_progress_bar import (
    LightProgressBar,
    StageProgressBar,
    TimeEstimator,
    dict_to_multi_dict,
    format_status,
    get_width,
    view_status,
)


def test_status_process(progress_bar_status_message: Dict[str, Any]):
    formatted_status = format_status(progress_bar_status_message)
    multi_status = dict_to_multi_dict(formatted_status)
    _ = view_status(multi_status)


def test_stage_progress_bar(progress_bar_status_message: Dict[str, Any]):
    progress_bar = StageProgressBar(get_width)
    progress_bar.update(progress_bar_status_message)
    progress_bar.finalize()


def test_time_estimator(eta_threshold: float = 0.0001):
    time_estimator = TimeEstimator(eta_threshold)
    time_estimator.reset()
    time_estimator.update(eta_threshold + 1)
    _ = str(time_estimator)


def test_light_progress_bar(progress_bar_status_message: Dict[str, Any]):
    light_progress_bar = LightProgressBar()
    light_progress_bar.pbar.update(progress_bar_status_message)
