import logging
import random
import typing as t

from ..base_module import BaseModule

__all__ = ["ValueGenerator"]


class ValueGenerator(BaseModule):
    _logger = logging.getLogger(__name__)

    def _get_next_float_value(self, min_value: float, max_value: float) -> float:
        return random.uniform(min_value, max_value)

    def _get_next_int_value(self, min_value: int, max_value: int) -> int:
        return random.randint(min_value, max_value)

    def _get_next_categorical_value(self, choices: t.List[t.Any]) -> t.Any:
        return random.choice(choices)

    def run(self, dtype: str, *args, **kwargs):
        if dtype == "float":
            return self._get_next_float_value(*args, **kwargs)
        if dtype == "int":
            return self._get_next_int_value(*args, **kwargs)
        if dtype == "category":
            return self._get_next_categorical_value(*args, **kwargs)

        raise ValueError(f"Did not expect dtype {dtype}")
