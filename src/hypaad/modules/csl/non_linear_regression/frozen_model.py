import logging
import typing as t

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

__all__ = ["NonLinearRegressionFrozenModel"]


class NonLinearRegressionFrozenModel:

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        transforms: t.Dict[str, t.Callable],
        base_model: LinearRegression,
        column_order: t.List[str],
        min_value: t.Optional[float] = None,
        max_value: t.Optional[float] = None,
        dtype: t.Optional[str] = None,
    ) -> None:
        self.transforms = transforms
        self.base_model = base_model
        self.min_value = min_value
        self.max_value = max_value
        self.column_order = column_order
        self.dtype = dtype

    def to_json(self) -> t.Dict:
        return {
            "transforms": self.transforms,
            "base_model": self.base_model.to_json(),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "column_order": self.column_order,
            "dtype": self.dtype,
        }

    def predict(self, data: pd.DataFrame) -> np.array:
        missing_columns = set(self.column_order) - set(data.columns)
        additional_columns = set(data.columns) - set(self.column_order)

        if missing_columns:
            raise ValueError(
                f"Expected data to exactly match columns {self.column_order}, but missing: {missing_columns}"
            )
        if additional_columns:
            self._logger.warning(
                "Expected data to exactly match columns %s, but additional: %s. Ignoring additional columns.",
                self.column_order,
                additional_columns,
            )

        data_transformed = data[self.column_order]
        for column in data_transformed.columns:
            data_transformed.loc[:, column] = self.transforms.get(column)(
                data_transformed[column]
            )

        result = self.base_model.predict(data_transformed)
        if self.min_value is not None or self.max_value is not None:
            result = np.clip(
                result,
                a_min=self.min_value,
                a_max=self.max_value,
            )
        result = pd.Series(result)

        if self.dtype is None:
            return result

        if self.dtype == "float64":
            return result

        if self.dtype == "int":
            return result.round(0).astype(int)

        raise ValueError(f"Unknown dtype: {self.dtype}")

    def predict_single(self, **kwargs) -> t.Any:
        data = pd.DataFrame({k: [v] for k, v in kwargs.items()})
        result = self.predict(data)
        return result[0]
