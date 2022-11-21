import logging
import shutil
import typing as t
from pathlib import Path
from time import sleep
from uuid import uuid4

import dask
import dask.bag
import numpy as np
import optuna
import pandas as pd
import requests
import timeeval
from dask.distributed import get_worker
from distributed import Client, get_client, rejoin, secede, worker_client
from optuna.storages import RedisStorage
from sklearn import metrics

# pylint: disable=cyclic-import
import hypex
from hypex.algorithms.algorithm_executor import AlgorithmRuntimeException
from hypex.trial_result import TrialResult

__all__ = ["Optimizer"]

REDIS_DB = "optuna"
REMOTE_DATA_DIR = "data"


class TrialIsFinishedException(Exception):
    pass


class Optimizer:
    def __init__(
        self,
        algorithm: str,
        storage: "hypex.OptunaStorage",
        study_name: str,
        optuna_study_name: str,
        timeseries_names: t.List[str],
        data_paths: t.Dict[str, t.Dict[str, Path]],
        parameter_distribution: hypex.MultidimensionalParameterDistribution,
        postprocess_parameter_guess: t.Optional[t.Callable] = None,
        registry: t.Optional["hypex.Registry"] = None,
    ):
        self._logger = self.__class__.get_logger()
        self.algorithm = algorithm
        self.storage = storage
        self.study_name = study_name
        self.optuna_study_name = optuna_study_name
        self.parameter_distribution = parameter_distribution
        self.postprocess_parameter_guess = postprocess_parameter_guess
        self.timeseries_names = timeseries_names
        self.data_paths = data_paths

        self.registry = registry
        self.trial_results: t.List[hypex.TrialResult] = []

        if self.registry is None:
            self._logger.info(
                "No custom registry was provided. Thus using the default registry"
            )
            self.registry = hypex.Registry.default()

    @classmethod
    def get_logger(cls):
        suffix = "[Not a dask worker]"
        try:
            suffix = get_worker().address
        except ValueError:
            pass
        return logging.getLogger(f"{cls.__name__} {suffix}")

    @classmethod
    def _run_algorithm(
        cls,
        dataset_path: str,
        results_dir: Path,
        params: t.Dict[str, t.Any],
        algorithm: str,
        registry: "hypex.Registry",
    ) -> np.ndarray:
        executor = registry.get_algorithm(algorithm)

        args = {
            "hyper_params": params,
            "results_path": results_dir,
            "resource_constraints": timeeval.ResourceConstraints(task_cpu_limit=1),
        }

        anomaly_scores = executor.execute(dataset_path=dataset_path, args=args)
        cls.get_logger().info("Received anomaly scores")
        return anomaly_scores

    @classmethod
    def run_and_score_algorithm(
        cls,
        trial_id: int,
        study_name: str,
        optuna_study_name: str,
        algorithm: str,
        params: t.Dict[str, t.Any],
        optuna_guess_params: t.Dict[str, t.Any],
        data_paths: t.Dict[str, t.Dict[str, Path]],
        timeseries_name: str,
        registry: "hypex.Registry",
    ) -> "hypex.TrialResult":
        test_dataset_path = data_paths[timeseries_name]["unsupervised"]
        test_is_anomaly = pd.read_csv(test_dataset_path)["is_anomaly"]

        # pylint: disable=fixme
        # TODO: supervised and semi-supervised detectors

        # Evaluate the algorithm's performance on the test dataset
        results_base_dir = Path("/tmp/hypex-anomaly-scores")
        results_base_dir.mkdir(exist_ok=True)

        results_dir = results_base_dir / str(uuid4())
        results_dir.mkdir(exist_ok=True)
        try:
            anomaly_scores = cls._run_algorithm(
                dataset_path=test_dataset_path,
                results_dir=results_dir,
                params=params,
                algorithm=algorithm,
                registry=registry,
            )
        except AlgorithmRuntimeException as e:
            cls.get_logger().error(e)
            return hypex.TrialResult(
                study_name=study_name,
                optuna_study_name=optuna_study_name,
                id=trial_id,
                worker=get_worker().name,
                algorithm=algorithm,
                timeseries=timeseries_name,
                optuna_guess_params=optuna_guess_params,
                params=params,
                auc_pr_score=np.nan,
                roc_auc_score=np.nan,
                best_threshold=np.nan,
                f1_score=np.nan,
                accuracy_score=np.nan,
                anomaly_scores_path="NOT_SAVED",
                exception=e,
            )

        cls.get_logger().info("Now removing the content in %s", str(results_dir))
        shutil.rmtree(results_dir)
        cls.get_logger().info("Sucessfully removed the content in %s", str(results_dir))

        score_path = "NOT_SAVED"
        # self._logger.info("Writing anomaly scores to disk...")
        # score_path = (
        #     trial_results_path
        #     / f"{study.name}__trial-{trial.number}__scores.csv"
        # )
        # pd.DataFrame({"anomaly_scores": anomaly_scores}).to_csv(
        #     score_path, index=False
        # )
        # self._logger.info(
        #     "Successfully written anomaly scores to %s", score_path
        # )

        any_is_nan = any(np.isnan(anomaly_scores))
        any_is_inf = any(np.isinf(anomaly_scores))

        if any_is_nan or any_is_inf:
            msg = f"Anomaly scores contain NaN={any_is_nan} or Inf={any_is_inf}. Marking trial as failed. algorithm={algorithm} params={params} timeseries={timeseries_name}"
            cls.get_logger().error(msg)
            return hypex.TrialResult(
                study_name=study_name,
                optuna_study_name=optuna_study_name,
                id=trial_id,
                worker=get_worker().name,
                algorithm=algorithm,
                timeseries=timeseries_name,
                optuna_guess_params=optuna_guess_params,
                params=params,
                auc_pr_score=np.nan,
                roc_auc_score=np.nan,
                best_threshold=np.nan,
                f1_score=np.nan,
                accuracy_score=np.nan,
                anomaly_scores_path="NOT_SAVED",
                exception=AlgorithmRuntimeException(msg),
            )

        # Calculate AUC_PR
        cls.get_logger().info("Now calculating the precision-recall-curve")
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_true=test_is_anomaly, probas_pred=anomaly_scores
        )
        cls.get_logger().info("Now calculating the AUC-PR score")
        auc_pr_score = metrics.auc(recall, precision)

        f1_scores = (2 * precision * recall) / (precision + recall)

        # Caluclate ROC
        cls.get_logger().info("Now calculating the ROC-AUC score")
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
        cls.get_logger().info("Now finding the best threshold based on the F1 score")
        idx = np.argmax(f1_scores)
        best_threshold = thresholds[idx]

        y_pred = anomaly_scores >= best_threshold
        cls.get_logger().info("Now calculating the accuracy score")
        accuracy_score = metrics.accuracy_score(y_true=test_is_anomaly, y_pred=y_pred)
        f1_score = metrics.f1_score(y_true=test_is_anomaly, y_pred=y_pred)

        # Store the trial's result
        cls.get_logger().info("Now saving the TrialResult")
        return hypex.TrialResult(
            study_name=study_name,
            optuna_study_name=optuna_study_name,
            id=trial_id,
            worker=get_worker().name,
            algorithm=algorithm,
            timeseries=timeseries_name,
            optuna_guess_params=optuna_guess_params,
            params=params,
            auc_pr_score=auc_pr_score,
            roc_auc_score=roc_auc_score,
            best_threshold=best_threshold,
            f1_score=f1_score,
            accuracy_score=accuracy_score,
            anomaly_scores_path=score_path,
        )

    def check_trial_is_updatable(self, trial_number: int) -> bool:
        storage = self.storage.get_storage_backend()
        study_id = storage.get_study_id_from_name(self.optuna_study_name)
        trial_id = storage.get_trial_id_from_study_id_trial_number(
            study_id=study_id,
            trial_number=trial_number,
        )
        if storage.get_trial(trial_id).state.is_finished():
            self._logger.warn("Trial %d is finished already", trial_id)
            raise TrialIsFinishedException(trial_id)

    def _create_objective(self) -> t.Callable[[optuna.Trial], float]:
        if len(self.trial_results) > 0:
            raise ValueError(
                "There are already trial results available. "
                "Please ensure to create a new Optimizer for each study."
            )

        # trial_results_path = Path("trial_results")
        # trial_results_path.mkdir(exist_ok=True)

        def func(trial: optuna.Trial):
            self.check_trial_is_updatable(trial_number=trial.number)

            # Generate a next best parameter guess
            self._logger.info("Now awaiting the next parameter guess")
            optuna_guess_params = self.parameter_distribution.next_guess(
                trial=trial,
            )
            self._logger.info("Parameter guess from Optuna: %s", optuna_guess_params)

            def _get_next_guess(
                timeseries_name: str,
            ) -> t.Tuple[t.Dict[str, t.Any], t.Dict[str, t.Any]]:
                params = (
                    self.postprocess_parameter_guess(
                        timeseries_name=timeseries_name, **optuna_guess_params
                    )
                    if self.postprocess_parameter_guess is not None
                    else optuna_guess_params
                )
                self._logger.info("Final params after postprocessing: %s", params)
                return params, optuna_guess_params

            trial_result: hypex.TrialResult = None
            if len(self.timeseries_names) == 1:
                timeseries_name = self.timeseries_names[0]
                params, optuna_guess_params = _get_next_guess(
                    timeseries_name=timeseries_name
                )
                trial_result = self.run_and_score_algorithm(
                    trial_id=trial.number,
                    study_name=self.study_name,
                    optuna_study_name=self.optuna_study_name,
                    algorithm=self.algorithm,
                    params=params,
                    optuna_guess_params=optuna_guess_params,
                    registry=self.registry,
                    data_paths=self.data_paths,
                    timeseries_name=timeseries_name,
                )
            else:

                def _run_and_score_algorithm(timeseries_names: t.List[str]):
                    if len(timeseries_names) != 1:
                        raise ValueError("Only one timeseries name allowed")

                    timeseries_name = timeseries_names[0]
                    params, optuna_guess_params = _get_next_guess(
                        timeseries_name=timeseries_name
                    )
                    return self.run_and_score_algorithm(
                        trial_id=trial.number,
                        study_name=self.study_name,
                        optuna_study_name=self.optuna_study_name,
                        algorithm=self.algorithm,
                        params=params,
                        optuna_guess_params=optuna_guess_params,
                        registry=self.registry,
                        data_paths=self.data_paths,
                        timeseries_name=timeseries_name,
                    )

                with worker_client() as client:
                    trial_group: t.List[hypex.TrialResult] = (
                        dask.bag.from_sequence(
                            self.timeseries_names,
                            npartitions=len(self.timeseries_names),
                        )
                        .map_partitions(
                            _run_and_score_algorithm,
                        )
                        .compute(scheduler=client)
                    )
                    # _future = client.submit(trial_group)
                    # trial_group = client.gather(
                    #     _future
                    # )

                # trial_group: t.List[hypex.TrialResult] = []
                # # TODO: submit as individual tasks
                # for timeseries_name in self.timeseries_names:
                #     trial_group.append(
                #         self.run_and_score_algorithm(
                #             trial_id=trial.number,
                #             study_name=self.study_name,
                #             optuna_study_name=self.optuna_study_name,
                #             algorithm=self.algorithm,
                #             params=params,
                #             optuna_guess_params=_params,
                #             registry=self.registry,
                #             data_paths=self.data_paths,
                #             timeseries_name=timeseries_name,
                #         )
                #     )
                mean_auc_pr_score = float(
                    np.mean([x.auc_pr_score for x in trial_group])
                )
                mean_roc_auc_score = float(
                    np.mean([x.roc_auc_score for x in trial_group])
                )
                mean_f1_score = float(np.mean([x.f1_score for x in trial_group]))
                mean_accuracy_score = float(
                    np.mean([x.accuracy_score for x in trial_group])
                )
                mean_best_threshold = float(
                    np.mean([x.best_threshold for x in trial_group])
                )

                trial_result = hypex.TrialResult(
                    study_name=self.study_name,
                    optuna_study_name=self.optuna_study_name,
                    id=trial.number,
                    worker=get_worker().name,
                    algorithm=self.algorithm,
                    timeseries="|".join(self.timeseries_names),
                    optuna_guess_params=optuna_guess_params,
                    params={},
                    auc_pr_score=mean_auc_pr_score,
                    roc_auc_score=mean_roc_auc_score,
                    best_threshold=mean_best_threshold,
                    f1_score=mean_f1_score,
                    accuracy_score=mean_accuracy_score,
                    anomaly_scores_path="NONE",
                    group_items=trial_group,
                )

            self.trial_results.append(trial_result)

            self.check_trial_is_updatable(trial_number=trial.number)

            self._logger.info("Now returning the AUC-PR score to the optimizer")
            return trial_result.auc_pr_score

        return func

    def _run(self, n_trials: int):
        optuna_study = optuna.load_study(
            study_name=self.optuna_study_name,
            storage=self.storage.get_storage_backend(),
        )
        self._logger.info(
            "Now running the optimization procedure for %d steps...", n_trials
        )
        obj = self._create_objective()
        self._logger.info("Created the objective")
        try:
            optuna_study.optimize(
                func=obj,
                n_trials=n_trials,
                catch=(
                    hypex.AlgorithmRuntimeException,
                    requests.exceptions.ReadTimeout,
                    TrialIsFinishedException,
                ),
            )
        except RuntimeError as e:
            self._logger.error(e)

    def run(
        self,
        n_trials: int,
    ) -> t.List["hypex.TrialResult"]:
        self._run(n_trials=n_trials)
        return self.trial_results
