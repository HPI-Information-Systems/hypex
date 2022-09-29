import typing as t

import numpy as np
from timeeval.utils.window import ReverseWindowing

from .algorithm_executor import AlgorithmExecutor
from .timeeval_utils import get_max_anomaly_length

__all__ = ["stomp"]

# post-processing for stomp
def post_stomp(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 30)
    if window_size < 4:
        print(
            "WARN: window_size must be at least 4. Dynamically fixing it by setting window_size to 4"
        )
        window_size = 4
    return ReverseWindowing(window_size=window_size).fit_transform(scores)


def get_timeeval_params(
    gutentag_config: t.Dict[str, t.Any]
) -> t.List[t.Dict[str, t.Any]]:
    max_anomaly_length = get_max_anomaly_length(gutentag_config)
    return [
        {
            "anomaly_window_size": max_anomaly_length,
            "exclusion_zone": 0.5,
            "random_state": 42,
        }
    ]


def stomp() -> AlgorithmExecutor:
    return AlgorithmExecutor(
        image_name="ghcr.io/mschroederi/stomp:latest",
        postprocess=post_stomp,
        default_params={
            "anomaly_window_size": 30,
            "exclusion_zone": 0.5,
            "random_state": 42,
        },
        get_timeeval_params=get_timeeval_params,
    )
