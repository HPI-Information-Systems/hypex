import math
import typing as t

import numpy as np
from timeeval.utils.window import ReverseWindowing

from .algorithm_executor import AlgorithmExecutor
from .timeeval_utils import get_dataset_period_size

__all__ = ["donut"]

# post-processing for Donut
def _post_donut(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 120)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)


def get_timeeval_params(
    gutentag_config: t.Dict[str, t.Any]
) -> t.List[t.Dict[str, t.Any]]:
    window_size = math.ceil(get_dataset_period_size(gutentag_config, default=120))
    return [
        {
            "window_size": window_size,
            "latent_size": 5,
            "regularization": 0.001,
            "linear_hidden_size": 130,
            "epochs": 500,
            "random_state": 42,
        }
    ]


def donut() -> AlgorithmExecutor:
    return AlgorithmExecutor(
        image_name="registry.gitlab.hpi.de/akita/i/donut:latest",
        postprocess=_post_donut,
        default_params={
            "window_size": 120,
            "latent_size": 5,
            "regularization": 0.001,
            "linear_hidden_size": 100,
            "epochs": 256,
            "random_state": 42,
        },
        get_timeeval_params=get_timeeval_params,
    )
