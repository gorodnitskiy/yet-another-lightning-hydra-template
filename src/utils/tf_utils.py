import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


def serialize_tf_events(tf_events_file_path: str) -> Any:
    if int(tf.__version__.split(".")[0]) == 2:
        from tensorflow.core.util import event_pb2

        for serialized_event in tf.data.TFRecordDataset(tf_events_file_path):
            event = event_pb2.Event.FromString(serialized_event.numpy())
            yield event
    else:
        from tensorflow.python.summary.summary_iterator import summary_iterator

        for event in summary_iterator(tf_events_file_path):
            yield event


def load_tf_events(
    tf_events_file_path: str, names: List[str]
) -> Dict[str, Any]:
    traces = defaultdict(list)
    for event in serialize_tf_events(tf_events_file_path):
        for value in event.summary.value:
            for name in names:
                if value.tag == name:
                    traces[name].append(value.simple_value)

    return {name: np.array(value) for name, value in traces.items()}


def load_metrics(
    tf_events_file_path: str,
    lead_trace: str,
    sub_traces: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    traces_list = [lead_trace]
    if sub_traces:
        traces_list += sub_traces
    traces = load_tf_events(tf_events_file_path, traces_list)

    lead_trace_value = traces[lead_trace]
    idx = lead_trace_value.argmax()
    if verbose:
        print(
            f"lead_trace: {lead_trace}:",
            f"argmax idx: {idx} / {len(lead_trace_value)}, "
            f"argmax value: {lead_trace_value[idx]}",
        )
        if sub_traces:
            for trace_name in sub_traces:
                print(f"sub_trace: {trace_name}: {traces[trace_name][idx]}")

    return traces
