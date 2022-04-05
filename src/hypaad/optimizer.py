import logging
import typing as t
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import timeeval
from sklearn import metrics

# pylint: disable=cyclic-import
import hypaad

__all__ = ["Optimizer"]

REDIS_DB = "optuna"
REMOTE_DATA_DIR = "data"


class Optimizer:
    def __init__(
        self,
        data_paths: t.Dict[str, t.Dict[str, Path]],
        registry: t.Optional["hypaad.Registry"] = None,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.data_paths = data_paths
        self.registry = registry

        if self.registry is None:
            self._logger.info(
                "No custom registry was provided. Thus using the default registry"
            )
            self.registry = hypaad.Registry.default()

    def _run_algorithm(
        self, algorithm: str, dataset_path: str, params: t.Dict[str, t.Any]
    ) -> np.ndarray:
        executor = self.registry.get_algorithm(algorithm)

        args = {
            "hyper_params": params,
            "results_path": Path("./results"),
            "resource_constraints": timeeval.ResourceConstraints(
                task_cpu_limit=1
            ),
        }

        anomaly_scores = executor.execute(dataset_path=dataset_path, args=args)
        return anomaly_scores

    def _create_objective(
        self, study: "hypaad.Study"
    ) -> t.Callable[[optuna.Trial], float]:
        if len(study.timeseries) != 1:
            raise Exception(
                "Currently only a single timeseries per study is supported"
            )
        test_dataset_path = self.data_paths[study.timeseries[0]]["unsupervised"]

        test_is_anomaly = pd.read_csv(test_dataset_path)["is_anomaly"]

        def func(trial: optuna.Trial):
            # Generate a next best parameter guess
            params = study.next_parameter_guess(trial)

            # pylint: disable=fixme
            # TODO: supervised and semi-supervised detectors

            # Evaluate the algorithm's performance on the test dataset
            anomaly_scores = self._run_algorithm(
                algorithm=study.algorithm,
                dataset_path=test_dataset_path,
                params=params,
            )
            precision, recall, _ = metrics.precision_recall_curve(
                y_true=test_is_anomaly, probas_pred=anomaly_scores
            )
            return metrics.auc(recall, precision)

        return func

    def run(
        self,
        study: "hypaad.Study",
        redis_url: str,
        n_trials: int,
        optuna_study_name: str,
    ):
        optuna_study = optuna.load_study(
            study_name=optuna_study_name,
            storage=optuna.storages.RedisStorage(url=redis_url),
        )

        self._logger.info(
            "Now running the optimization procedure for %d steps...", n_trials
        )
        optuna_study.optimize(
            func=self._create_objective(study=study),
            n_trials=n_trials,
            catch=tuple([IndexError]),
        )
