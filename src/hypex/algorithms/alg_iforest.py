import typing as t

from .algorithm_executor import AlgorithmExecutor

__all__ = ["iforest"]


def get_timeeval_params(
    gutentag_config: t.Dict[str, t.Any]
) -> t.List[t.Dict[str, t.Any]]:
    param_suggestions = []
    for n_trees in [10, 100, 1000]:
        param_suggestions.append({"n_trees": n_trees, "random_state": 42})
    return param_suggestions


def iforest() -> AlgorithmExecutor:
    return AlgorithmExecutor(
        image_name="ghcr.io/mschroederi/iforest:latest",
        postprocess=None,
        default_params={
            "n_trees": 100,
            "random_state": 42,
        },
        get_timeeval_params=get_timeeval_params,
    )
