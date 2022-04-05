import typing as t

import numpy as np
from timeeval.utils.window import ReverseWindowing

from .algorithm_executor import AlgorithmExecutor

__all__ = ["series2graph"]

# post-processing for s2g
def post_s2g(scores: np.ndarray, args: t.Dict[str, t.Any]) -> np.ndarray:
    hyper_params = args.get("hyper_params", {})
    window_size = hyper_params.get("window_size", 50)
    query_window_size = hyper_params.get("query_window_size", 75)
    convolution_size = hyper_params.get("convolution_size", window_size // 3)
    size = (window_size + convolution_size) + query_window_size + 4
    return ReverseWindowing(window_size=size).fit_transform(scores)


def series2graph() -> AlgorithmExecutor:
    return AlgorithmExecutor(
        image_name="sopedu:5000/akita/series2graph",
        postprocess=post_s2g,
    )
