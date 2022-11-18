import typing as t

import numpy as np
import pandas as pd

import hypex

__all__ = []


class NonLinearIndependenceTest:
    @classmethod
    def independence_test(
        cls, df: pd.DataFrame, x: str, y: str, sample_weight: np.array = None
    ) -> "hypex.RegressionResult":
        return hypex.NonLinearRegression.fit(
            data_x=df[[x]],
            data_y=df[y],
            sample_weight=sample_weight,
            keep_dtype=True,
        )

    @classmethod
    def conditional_independence_test(
        cls,
        df: pd.DataFrame,
        x: str,
        y: str,
        z: t.Set[str],
        sample_weight: np.array,
    ) -> "hypex.RegressionResult":
        data_x = df[x]
        data_y = df[y]
        data_z = df[list(z)]

        model_ZX = hypex.NonLinearRegression.fit(
            data_x=data_z,
            data_y=data_x,
            sample_weight=sample_weight,
            keep_dtype=False,
        ).model
        residual_ZX = data_x - model_ZX.predict(data_z)

        model_ZY = hypex.NonLinearRegression.fit(
            data_x=data_z,
            data_y=data_y,
            sample_weight=sample_weight,
            keep_dtype=False,
        ).model
        residual_ZY = data_y - model_ZY.predict(data_z)

        result = hypex.NonLinearRegression.fit(
            data_x=pd.DataFrame({"residual_ZX": residual_ZX}),
            data_y=residual_ZY,
            sample_weight=sample_weight,
            keep_dtype=False,
        )

        return result
