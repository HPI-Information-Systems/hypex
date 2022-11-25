import typing as t

from .algorithm_executor import AlgorithmExecutor

__all__ = ["pst"]


def get_timeeval_params(
    gutentag_config: t.Dict[str, t.Any]
) -> t.List[t.Dict[str, t.Any]]:
    return [
        {
            "window_size": 5,
            "max_depth": 4,
            "n_min": 1,
            "n_bins": 5,
            "random_state": 42,
        }
    ]


def pst() -> AlgorithmExecutor:
    return AlgorithmExecutor(
        image_name="registry.gitlab.hpi.de/akita/i/pst:latest",
        postprocess=None,
        default_params={
            "window_size": 5,
            "max_depth": 4,
            "n_min": 1,
            "n_bins": 5,
            "random_state": 42,
        },
        get_timeeval_params=get_timeeval_params,
    )
