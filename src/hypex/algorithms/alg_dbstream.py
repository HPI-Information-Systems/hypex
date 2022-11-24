import math
import typing as t

from .algorithm_executor import AlgorithmExecutor
from .timeeval_utils import get_dataset_period_size

__all__ = ["dbstream"]


def get_timeeval_params(
    gutentag_config: t.Dict[str, t.Any]
) -> t.List[t.Dict[str, t.Any]]:
    parameter_suggestions = []
    dataset_period_size = math.ceil(
        get_dataset_period_size(gutentag_config, default=20)
    )
    for alpha in [0.01, 0.05, 0.1, 0.25, 0.5]:
        for n_clusters in [5, 10, 30, 40, 50]:
            radius_default_value = 0.1
            for radius in [
                0.7 * radius_default_value,
                radius_default_value,
                1.3 * radius_default_value,
            ]:
                for _lambda in [0.0001, 0.001, 0.01, 0.1]:
                    for min_weight in [0, 0.2, 0.5, 1]:
                        for shared_density in [True, False]:
                            parameter_suggestions.append(
                                {
                                    "window_size": dataset_period_size,
                                    "radius": radius,
                                    "lambda": _lambda,
                                    "n_clusters": n_clusters,
                                    "alpha": alpha,
                                    "min_weight": min_weight,
                                    "shared_density": shared_density,
                                    "random_state": 42,
                                }
                            )
    return parameter_suggestions


def dbstream() -> AlgorithmExecutor:
    return AlgorithmExecutor(
        image_name="registry.gitlab.hpi.de/akita/i/dbstream:latest",
        postprocess=None,
        default_params={
            "window_size": 20,
            "radius": 0.1,
            "lambda": 0.001,
            "n_clusters": 0,
            "alpha": 0.1,
            "min_weight": 0.0,
            "random_state": 42,
        },
        get_timeeval_params=get_timeeval_params,
    )
