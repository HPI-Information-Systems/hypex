import typing as t

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, hstack

from .algorithm_executor import AlgorithmExecutor
from .timeeval_utils import get_max_anomaly_length

__all__ = ["grammarviz3"]

# post-processing for GrammarViz
def post_grammarviz(scores: np.ndarray, args: dict) -> np.ndarray:
    results = pd.DataFrame(scores, columns=["index", "score", "length"])
    results = results.set_index("index")
    anomalies = results[results["score"] > 0.0]

    # use scipy sparse matrix to save memory
    matrix = csc_matrix((len(results), 1), dtype=np.float64)
    counts = np.zeros(len(results))
    for i, row in anomalies.iterrows():
        idx = int(row.name)
        length = int(row["length"])
        tmp = np.zeros(len(results))
        tmp[idx : idx + length] = np.repeat([row["score"]], repeats=length)
        tmp = tmp.reshape(-1, 1)
        matrix = hstack([matrix, tmp])
        counts[idx : idx + length] += 1
    sums = matrix.sum(axis=1)
    counts = counts.reshape(-1, 1)
    scores = np.zeros_like(sums)
    np.divide(sums, counts, out=scores, where=counts != 0)
    # returns the completely flattened array (from `[[1.2], [2.3]]` to `[1.2, 2.3]`)
    return scores.A1


def get_timeeval_params(
    gutentag_config: t.Dict[str, t.Any]
) -> t.List[t.Dict[str, t.Any]]:
    param_suggestions = []
    max_anomaly_length = get_max_anomaly_length(gutentag_config)

    for alphabet_size in [3, 4, 5, 6]:
        for paa_transform_size in [3, 4, 5]:
            param_suggestions.append(
                {
                    "anomaly_window_size": max_anomaly_length,
                    "paa_transform_size": paa_transform_size,
                    "alphabet_size": alphabet_size,
                    "normalization_threshold": 0.01,
                    "random_state": 42,
                }
            )
    return param_suggestions


def grammarviz3() -> AlgorithmExecutor:
    return AlgorithmExecutor(
        image_name="ghcr.io/mschroederi/grammarviz3:latest",
        postprocess=post_grammarviz,
        default_params={
            "anomaly_window_size": 170,
            "paa_transform_size": 4,
            "alphabet_size": 4,
            "normalization_threshold": 0.01,
            "random_state": 42,
        },
        get_timeeval_params=get_timeeval_params,
    )
