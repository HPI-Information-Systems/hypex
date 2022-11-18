import random

import numpy as np

__all__ = ["use_seed"]


def use_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
