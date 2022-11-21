import math
import typing as t
from email.policy import default

import numpy as np
from timeeval.utils.window import ReverseWindowing

from .algorithm_executor import AlgorithmExecutor

__all__ = ["torsk"]

# post-processing for Torsk
def _post_torsk(scores: np.ndarray, args: dict) -> np.ndarray:
    pred_size = args.get("hyper_params", {}).get("prediction_window_size", 20)
    context_window_size = args.get("hyper_params", {}).get("context_window_size", 10)
    size = pred_size * context_window_size + 1
    is_anomaly = ReverseWindowing(window_size=size).fit_transform(scores)
    return np.nan_to_num(is_anomaly, nan=0)


def get_timeeval_params(
    gutentag_config: t.Dict[str, t.Any]
) -> t.List[t.Dict[str, t.Any]]:
    parameter_suggestions = []
    default_spectral_radius = 2.0

    for train_window_size in [50, 100, 500, 1000]:
        for prediction_window_size in [1, 5, 10, 50]:
            for context_window_size in [5, 10, 30, 40, 50]:
                for spectral_radius in [
                    0.7 * default_spectral_radius,
                    default_spectral_radius,
                    1.3 * default_spectral_radius,
                ]:
                    for input_map_size in [50, 100, 500]:

                        parameter_suggestions.append(
                            {
                                "input_map_size": input_map_size,
                                "input_map_scale": 0.125,
                                "context_window_size": context_window_size,
                                "train_window_size": train_window_size,
                                "prediction_window_size": prediction_window_size,
                                "transient_window_size": math.ceil(
                                    0.2 * train_window_size
                                ),
                                "spectral_radius": spectral_radius,
                                "density": 0.01,
                                "scoring_small_window_size": 10,
                                "scoring_large_window_size": 100,
                                "random_state": 42,
                            }
                        )
    return parameter_suggestions


def torsk() -> AlgorithmExecutor:
    return AlgorithmExecutor(
        image_name="ghcr.io/mschroederi/torsk:latest",
        postprocess=_post_torsk,
        default_params={
            "input_map_size": 100,
            "input_map_scale": 0.125,
            "context_window_size": 10,
            "train_window_size": 50,
            "prediction_window_size": 20,
            "transient_window_size": 10,
            "spectral_radius": 2.0,
            "density": 0.01,
            "scoring_small_window_size": 10,
            "scoring_large_window_size": 100,
            "random_state": 42,
        },
        get_timeeval_params=get_timeeval_params,
    )
