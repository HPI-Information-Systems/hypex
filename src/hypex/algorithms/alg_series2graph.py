import math
import typing as t

import numpy as np
from timeeval.utils.window import ReverseWindowing

from .algorithm_executor import AlgorithmExecutor
from .timeeval_utils import get_dataset_period_size

__all__ = ["series2graph"]

# post-processing for s2g
def post_s2g(scores: np.ndarray, args: t.Dict[str, t.Any]) -> np.ndarray:
    hyper_params = args.get("hyper_params", {})
    window_size = hyper_params.get("window_size", 50)
    query_window_size = hyper_params.get("query_window_size", 75)
    convolution_size = hyper_params.get("convolution_size", window_size // 3)
    size = (window_size + convolution_size) + query_window_size + 4
    return ReverseWindowing(window_size=size).fit_transform(scores)


def get_timeeval_params(
    gutentag_config: t.Dict[str, t.Any]
) -> t.List[t.Dict[str, t.Any]]:
    dataset_period_size = get_dataset_period_size(gutentag_config, default=50)
    window_size = math.ceil(1.0 * dataset_period_size)
    return [
        {
            "window_size": window_size,
            "query_window_size": math.ceil(1.5 * window_size),
            "rate": 100,
            "random_state": 42,
        }
    ]


def series2graph() -> AlgorithmExecutor:
    return AlgorithmExecutor(
        image_name="ghcr.io/mschroederi/series2graph:latest",
        postprocess=post_s2g,
        default_params={
            "window_size": 50,
            "query_window_size": 75,
            "rate": 30,
            "random_state": 42,
        },
        get_timeeval_params=get_timeeval_params,
    )
