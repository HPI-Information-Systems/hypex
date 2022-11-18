import abc
import json
import logging
from pathlib import Path

import pandas as pd

__all__ = []


class BaseResult(abc.ABC):
    @classmethod
    def get_logger(self):
        return logging.getLogger(__name__)

    def _save_dict(self, data: dict, path: Path):
        self.get_logger().info("Saving dict to %s", path)
        with open(path, "w") as file:
            json.dump(data, file)

    @classmethod
    def _load_dict(self, path: Path):
        self.get_logger().info("Loading dict from %s", path)
        with open(path, "r") as file:
            return json.load(file)

    def _save_dataframe(self, data: pd.DataFrame, path: Path):
        self.get_logger().info("Saving DataFrame to %s", path)
        data.to_csv(path, index=False)

    @classmethod
    def _load_dataframe(self, path: Path):
        self.get_logger().info("Loading DataFrame from %s", path)
        return pd.read_csv(path)

    def save(self, output_dir: Path) -> None:
        raise NotImplementedError()
