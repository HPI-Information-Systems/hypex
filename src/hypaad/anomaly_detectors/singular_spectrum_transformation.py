import banpei
import numpy as np

from hypaad.anomaly_detectors.base_executor import BaseExecutor

__all__ = ["SSTExecutor"]


class SSTExecutor(BaseExecutor):
    model: banpei.SST

    def __init__(
        self,
        window_size: int,
        num_vectors: int,
        num_columns: int,
        lag_time: int,
    ):
        super().__init__()
        self.model = banpei.SST(
            w=window_size, m=num_vectors, k=num_columns, L=lag_time
        )

    def fit(self, data: np.array):
        pass

    def predict(self, data: np.array) -> np.array:
        y_pred = self.model.detect(data[:, 1])
        y_pred[y_pred == -1] = 0
        return y_pred
