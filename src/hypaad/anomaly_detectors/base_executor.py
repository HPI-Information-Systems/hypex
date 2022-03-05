import abc

import numpy as np

__all__ = ["BaseExecutor"]


class BaseExecutor(abc.ABC):
    @abc.abstractmethod
    def fit(self, data: np.array):
        pass

    @abc.abstractmethod
    def predict(self, data: np.array):
        pass
