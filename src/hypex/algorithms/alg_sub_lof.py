import math
import typing as t

import numpy as np
from timeeval.utils.window import ReverseWindowing

from .algorithm_executor import AlgorithmExecutor
from .timeeval_utils import get_dataset_period_size

__all__ = ["sub_lof"]


def postprocess_sub_lof(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 100)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)


def get_timeeval_params(
    gutentag_config: t.Dict[str, t.Any]
) -> t.List[t.Dict[str, t.Any]]:
    dataset_period_size = math.ceil(
        get_dataset_period_size(gutentag_config, default=100)
    )
    return [
        {
            "window_size": dataset_period_size,
            "n_neighbors": 50,
            "leaf_size": 20,
            "distance_metric_order": 2,
            "random_state": 42,
        }
    ]


def sub_lof() -> AlgorithmExecutor:
    return AlgorithmExecutor(
        image_name="registry.gitlab.hpi.de/akita/i/subsequence_lof:latest",
        postprocess=postprocess_sub_lof,
        default_params={
            "window_size": 100,
            "n_neighbors": 20,
            "leaf_size": 30,
            "distance_metric_order": 2,
            "random_state": 42,
        },
        get_timeeval_params=get_timeeval_params,
    )
