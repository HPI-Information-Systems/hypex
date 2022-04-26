__all__ = ["EarlyStoppingCallback"]

import operator

import numpy as np
import optuna


# Note: Currently not used (requires more than 1 trial per execution)
class EarlyStoppingCallback:
    """Early stopping callback for Optuna."""

    def __init__(
        self, early_stopping_rounds: int, direction: str = "minimize"
    ) -> None:
        self.early_stopping_rounds = early_stopping_rounds

        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Do early stopping."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()
