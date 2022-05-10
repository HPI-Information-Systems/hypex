import logging
import typing as t
from pathlib import Path
from uuid import uuid4

import numpy as np
import optuna
import pandas as pd
import timeeval
from dask.distributed import get_worker
from sklearn import metrics

# pylint: disable=cyclic-import
import hypaad

__all__ = ["Optimizer"]

REDIS_DB = "optuna"
REMOTE_DATA_DIR = "data"


class Optimizer:
    def __init__(
        self,
        registry: t.Optional["hypaad.Registry"] = None,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.registry = registry
        self.trial_results: t.List[hypaad.TrialResult] = []

        if self.registry is None:
            self._logger.info(
                "No custom registry was provided. Thus using the default registry"
            )
            self.registry = hypaad.Registry.default()

    def _run_algorithm(
        self,
        algorithm: str,
        dataset_path: str,
        results_dir: Path,
        params: t.Dict[str, t.Any],
    ) -> np.ndarray:
        executor = self.registry.get_algorithm(algorithm)

        args = {
            "hyper_params": params,
            "results_path": results_dir,
            "resource_constraints": timeeval.ResourceConstraints(
                task_cpu_limit=1
            ),
        }

        anomaly_scores = executor.execute(dataset_path=dataset_path, args=args)
        return anomaly_scores

    def _create_objective(
        self,
        study: "hypaad.Study",
        data_paths: t.Dict[str, t.Dict[str, Path]],
    ) -> t.Callable[[optuna.Trial], float]:
        if len(study.timeseries) != 1:
            raise Exception(
                "Currently only a single timeseries per study is supported"
            )
        if len(self.trial_results) > 0:
            raise ValueError(
                "There are already trial results available. "
                "Please ensure to create a new Optimizer for each study."
            )
        test_dataset_path = data_paths[study.timeseries[0]]["unsupervised"]

        test_is_anomaly = pd.read_csv(test_dataset_path)["is_anomaly"]

        trial_results_path = Path("trial_results")
        trial_results_path.mkdir(exist_ok=True)

        def func(trial: optuna.Trial):
            # Generate a next best parameter guess
            params = study.next_parameter_guess(trial)

            # pylint: disable=fixme
            # TODO: supervised and semi-supervised detectors

            # Evaluate the algorithm's performance on the test dataset
            results_dir = Path("/tmp/hypaad-anomaly-scores")
            results_dir.mkdir(exist_ok=True)
            anomaly_scores = self._run_algorithm(
                algorithm=study.algorithm,
                dataset_path=test_dataset_path,
                results_dir=results_dir / str(uuid4()),
                params=params,
            )
            self._logger.info("Writing anomaly scores to disk...")
            score_path = (
                trial_results_path
                / f"{study.name}__trial-{trial.number}__scores.csv"
            )
            pd.DataFrame({"anomaly_scores": anomaly_scores}).to_csv(
                score_path, index=False
            )
            self._logger.info(
                "Successfully written anomaly scores to %s", score_path
            )

            # Calculate AUC_PR
            precision, recall, thresholds = metrics.precision_recall_curve(
                y_true=test_is_anomaly, probas_pred=anomaly_scores
            )
            auc_pr_score = metrics.auc(recall, precision)

            f1_scores = (2 * precision * recall) / (precision + recall)

            # Caluclate ROC
            roc_auc_score = metrics.roc_auc_score(
                y_true=test_is_anomaly, y_score=anomaly_scores
            )
            # fpr, tpr, thresholds = metrics.roc_curve(
            #     y_true=test_is_anomaly, y_score=anomaly_scores
            # )

            # # Caluclate the best threshold (based on tpr fpr)
            # idx = np.argmax(tpr - fpr)
            # best_threshold = thresholds[idx]

            # Caluclate the best threshold (based on f1 score)
            idx = np.argmax(f1_scores)
            best_threshold = thresholds[idx]

            y_pred = anomaly_scores >= best_threshold
            accuracy_score = metrics.accuracy_score(
                y_true=test_is_anomaly, y_pred=y_pred
            )
            f1_score = metrics.f1_score(y_true=test_is_anomaly, y_pred=y_pred)

            # Store the trial's result
            self.trial_results.append(
                hypaad.TrialResult(
                    study_name=study.name,
                    id=trial.number,
                    worker=get_worker().name,
                    algorithm=study.algorithm,
                    params=params,
                    auc_pr_score=auc_pr_score,
                    roc_auc_score=roc_auc_score,
                    best_threshold=best_threshold,
                    f1_score=f1_score,
                    accuracy_score=accuracy_score,
                    anomaly_scores_path=score_path,
                )
            )

            return auc_pr_score

        return func

    def run(
        self,
        study: "hypaad.Study",
        redis_url: str,
        n_trials: int,
        optuna_study_name: str,
        data_paths: t.Dict[str, t.Dict[str, Path]],
    ):
        optuna_study = optuna.load_study(
            study_name=optuna_study_name,
            storage=optuna.storages.RedisStorage(url=redis_url),
        )

        self._logger.info(
            "Now running the optimization procedure for %d steps...", n_trials
        )
        optuna_study.optimize(
            func=self._create_objective(study=study, data_paths=data_paths),
            callbacks=[hypaad.EarlyStoppingCallback],
            n_trials=n_trials,
        )
