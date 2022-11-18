import abc
import random
import typing as t

import numpy as np

import hypex

__all__ = ["BaseModule"]


class BaseModule(abc.ABC):
    def __init__(self, seed: int) -> None:
        self.seed = seed
        hypex.use_seed(seed=self.seed)

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> t.Any:
        raise NotImplementedError()
