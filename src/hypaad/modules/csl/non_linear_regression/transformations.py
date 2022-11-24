import numpy as np

__all__ = ["Transformations"]


class Transformations:
    @classmethod
    def linear(cls, x: np.array):
        return x

    @classmethod
    def hyperbola(cls, x: np.array, eps=1e-5):
        A = x.astype(float, copy=True)
        A[A == 0] = eps
        return 1 / A
