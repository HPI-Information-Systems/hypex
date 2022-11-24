import typing as t

from .algorithm_executor import AlgorithmExecutor

__all__ = ["dwt_mlead"]


def get_timeeval_params(
    gutentag_config: t.Dict[str, t.Any]
) -> t.List[t.Dict[str, t.Any]]:
    return [
        {
            "start_level": 3,
            "quantile_epsilon": 0.1,
            "random_state": 42,
        }
    ]


def dwt_mlead() -> AlgorithmExecutor:
    return AlgorithmExecutor(
        image_name="registry.gitlab.hpi.de/akita/i/dwt_mlead:latest",
        postprocess=None,
        default_params={
            "start_level": 3,
            "quantile_epsilon": 0.01,
            "random_state": 42,
        },
        get_timeeval_params=get_timeeval_params,
    )
