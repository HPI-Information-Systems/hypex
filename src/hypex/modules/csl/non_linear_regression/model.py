import itertools
import math
import typing as t
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import hypex

from .transformations import Transformations

__all__ = ["NonLinearRegression", "RegressionResult"]

transform_linear = ("linear", Transformations.linear)
transform_hyperbola = ("hyperbola", Transformations.hyperbola)
ALL_TRANSFORMATIONS = [transform_linear, transform_hyperbola]


@dataclass
class RegressionResult:
    transform_names: t.Dict[str, t.Any]
    transform_funcs: t.Dict[str, t.Callable[[np.ndarray], np.ndarray]]
    model: "hypex.NonLinearRegressionFrozenModel"
    score: float
    error: float

    def to_edge_data(self):
        return {
            "transforms": self.transform_names,
            "score": self.score,
            "error": self.error,
            "model_coef": self.model.base_model.coef_,
            "model_intercept": self.model.base_model.intercept_,
        }


class NonLinearRegression:
    @classmethod
    def _fit_linear_regression(
        cls,
        data_x: np.ndarray,
        data_y: np.array,
        min_value: t.Optional[float] = None,
        max_value: t.Optional[float] = None,
        sample_weight: t.Optional[np.array] = None,
    ) -> t.Tuple[LinearRegression, float, float]:
        model = LinearRegression().fit(
            X=data_x,
            y=data_y,
            sample_weight=sample_weight,
        )

        # Calculate the RMSE
        y_pred = model.predict(data_x)
        if min_value is not None or max_value is not None:
            y_pred = np.clip(y_pred, a_min=min_value, a_max=max_value)
        error = mean_squared_error(
            y_true=data_y,
            y_pred=y_pred,
            sample_weight=sample_weight,
            squared=False,
        )
        score = r2_score(data_y, y_pred, sample_weight=sample_weight)

        return (model, score, error)

    @classmethod
    def fit(
        cls,
        data_x: pd.DataFrame,
        data_y: pd.Series,
        min_value: t.Optional[float] = None,
        max_value: t.Optional[float] = None,
        sample_weight: t.Optional[pd.Series] = None,
        keep_dtype: bool = False,
    ) -> RegressionResult:
        result = None
        best_error = math.inf

        for transform_items in list(
            itertools.product(ALL_TRANSFORMATIONS, repeat=len(data_x.columns))
        ):
            _tmp = {
                col: transform
                for col, transform in zip(data_x.columns, transform_items)
            }
            transform_names = {k: v[0] for k, v in _tmp.items()}
            transform_funcs = {k: v[1] for k, v in _tmp.items()}
            data_X = pd.DataFrame(
                {
                    col: transform_funcs[col](data_x[col])
                    for col in data_x.columns
                }
            )

            linear_model, score, error = cls._fit_linear_regression(
                data_x=data_X,
                data_y=data_y,
                min_value=min_value,
                max_value=max_value,
                sample_weight=sample_weight,
            )

            if error < best_error:
                best_error = error
                model = hypex.NonLinearRegressionFrozenModel(
                    transforms=transform_funcs,
                    base_model=linear_model,
                    min_value=min_value,
                    max_value=max_value,
                    column_order=list(data_x.columns),
                    dtype=data_y.dtype if keep_dtype else None,
                )
                result = RegressionResult(
                    transform_names=transform_names,
                    transform_funcs=transform_funcs,
                    model=model,
                    score=score,
                    error=error,
                )

        # for transform_name, transform_func in ALL_TRANSFORMATIONS:
        #     data_X = transform_func(data_x)

        #     linear_model, score, error = cls._fit_linear_regression(
        #         data_x=data_X, data_y=data_y, sample_weight=sample_weight
        #     )

        #     if error < best_error:
        #         best_error = error
        #         model = hypex.NonLinearRegressionFrozenModel(
        #             transform=transform_func,
        #             base_model=linear_model,
        #             column_order=list(data_x.columns),
        #             dtype=data_y.dtype if keep_dtype else None,
        #         )
        #         result = RegressionResult(
        #             transform=transform_name,
        #             transform_func=transform_func,
        #             model=model,
        #             score=score,
        #             error=error,
        #         )
        return result
