import abc
import typing as t

import dask
import pandas as pd

import hypex

__all__ = []


class BaseRunner(abc.ABC):
    @classmethod
    def _trials_to_df(
        cls, trial_results: t.List["hypex.TrialResult"]
    ) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            data=[t.to_dict() for t in trial_results],
            orient="columns",
        )

    def __init__(self, seed: int, storage: "hypex.OptunaStorage") -> None:
        self.seed = seed
        self.storage = storage
        hypex.use_seed(seed=self.seed)

    def run(self):
        raise NotImplementedError()
