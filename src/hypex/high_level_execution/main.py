import logging
import time
import typing as t
from dataclasses import dataclass
from pathlib import Path

import dask
import dask.distributed
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tsfresh.feature_extraction.feature_calculators import (
    abs_energy,
    fourier_entropy,
    length,
    longest_strike_above_mean,
    longest_strike_below_mean,
    maximum,
    mean,
    median,
    minimum,
    number_peaks,
    standard_deviation,
    variance,
)

import hypex

from ..modules.base_result import BaseResult
from ..utils.utils import docker_prune, docker_prune_cleanup

__all__ = ["Main"]

DEFAULT_SEED = 2
DATASET_TEST_SIZE = 0.2
DATASET_VALIDATION_SIZE = 0.25
TRAIN_BATCH_SIZE = 50


def get_tsfresh_features(path: Path):
    FEATURES = [
        # basic statistics
        ("length", length),
        ("minimum", minimum),
        ("maximum", maximum),
        ("mean", mean),
        ("median", median),
        ("standard_deviation", standard_deviation),
        # frequency
        ("number_peaks", lambda x: number_peaks(x, n=20)),
        ("longest_strike_above_mean", longest_strike_above_mean),
        ("longest_strike_below_mean", longest_strike_below_mean),
        # noise
        ("variance", variance),
        # complexity
        ("fourier_entropy", lambda x: fourier_entropy(x, bins=100)),
        ("abs_energy", abs_energy),
    ]

    values = pd.read_csv(path).iloc[:, 1]

    return [{"name": name, "value": float(func(values))} for name, func in FEATURES]


base_data_path = Path(
    "/home/projects/akita/data/benchmark-data/data-processed/univariate/KDD-TSAD"
)
test_datasets: t.Dict[str, t.Dict[str, Path]] = {
    "229_UCR_Anomaly_mit14134longtermecg": {
        "unsupervised": base_data_path / "229_UCR_Anomaly_mit14134longtermecg.test.csv"
    },
    "122_UCR_Anomaly_ECG3": {
        "unsupervised": base_data_path / "122_UCR_Anomaly_ECG3.test.csv"
    },
    "235_UCR_Anomaly_mit14157longtermecg": {
        "unsupervised": base_data_path / "235_UCR_Anomaly_mit14157longtermecg.test.csv"
    },
    "016_UCR_Anomaly_DISTORTEDECG4": {
        "unsupervised": base_data_path / "016_UCR_Anomaly_DISTORTEDECG4.test.csv"
    },
    "233_UCR_Anomaly_mit14157longtermecg": {
        "unsupervised": base_data_path / "233_UCR_Anomaly_mit14157longtermecg.test.csv"
    },
    "230_UCR_Anomaly_mit14134longtermecg": {
        "unsupervised": base_data_path / "230_UCR_Anomaly_mit14134longtermecg.test.csv"
    },
    "125_UCR_Anomaly_ECG4": {
        "unsupervised": base_data_path / "125_UCR_Anomaly_ECG4.test.csv"
    },
    "222_UCR_Anomaly_mit14046longtermecg": {
        "unsupervised": base_data_path / "222_UCR_Anomaly_mit14046longtermecg.test.csv"
    },
    "012_UCR_Anomaly_DISTORTEDECG2": {
        "unsupervised": base_data_path / "012_UCR_Anomaly_DISTORTEDECG2.test.csv"
    },
    "126_UCR_Anomaly_ECG4": {
        "unsupervised": base_data_path / "126_UCR_Anomaly_ECG4.test.csv"
    },
    "228_UCR_Anomaly_mit14134longtermecg": {
        "unsupervised": base_data_path / "228_UCR_Anomaly_mit14134longtermecg.test.csv"
    },
    "123_UCR_Anomaly_ECG4": {
        "unsupervised": base_data_path / "123_UCR_Anomaly_ECG4.test.csv"
    },
    "231_UCR_Anomaly_mit14134longtermecg": {
        "unsupervised": base_data_path / "231_UCR_Anomaly_mit14134longtermecg.test.csv"
    },
    "223_UCR_Anomaly_mit14046longtermecg": {
        "unsupervised": base_data_path / "223_UCR_Anomaly_mit14046longtermecg.test.csv"
    },
    "014_UCR_Anomaly_DISTORTEDECG3": {
        "unsupervised": base_data_path / "014_UCR_Anomaly_DISTORTEDECG3.test.csv"
    },
    "017_UCR_Anomaly_DISTORTEDECG4": {
        "unsupervised": base_data_path / "017_UCR_Anomaly_DISTORTEDECG4.test.csv"
    },
    "096_UCR_Anomaly_NOISEECG4": {
        "unsupervised": base_data_path / "096_UCR_Anomaly_NOISEECG4.test.csv"
    },
    "087_UCR_Anomaly_DISTORTEDsel840mECG1": {
        "unsupervised": base_data_path / "087_UCR_Anomaly_DISTORTEDsel840mECG1.test.csv"
    },
    "225_UCR_Anomaly_mit14046longtermecg": {
        "unsupervised": base_data_path / "225_UCR_Anomaly_mit14046longtermecg.test.csv"
    },
    "013_UCR_Anomaly_DISTORTEDECG3": {
        "unsupervised": base_data_path / "013_UCR_Anomaly_DISTORTEDECG3.test.csv"
    },
    "226_UCR_Anomaly_mit14046longtermecg": {
        "unsupervised": base_data_path / "226_UCR_Anomaly_mit14046longtermecg.test.csv"
    },
    "227_UCR_Anomaly_mit14134longtermecg": {
        "unsupervised": base_data_path / "227_UCR_Anomaly_mit14134longtermecg.test.csv"
    },
    "119_UCR_Anomaly_ECG1": {
        "unsupervised": base_data_path / "119_UCR_Anomaly_ECG1.test.csv"
    },
    "224_UCR_Anomaly_mit14046longtermecg": {
        "unsupervised": base_data_path / "224_UCR_Anomaly_mit14046longtermecg.test.csv"
    },
    "120_UCR_Anomaly_ECG2": {
        "unsupervised": base_data_path / "120_UCR_Anomaly_ECG2.test.csv"
    },
    "232_UCR_Anomaly_mit14134longtermecg": {
        "unsupervised": base_data_path / "232_UCR_Anomaly_mit14134longtermecg.test.csv"
    },
    "011_UCR_Anomaly_DISTORTEDECG1": {
        "unsupervised": base_data_path / "011_UCR_Anomaly_DISTORTEDECG1.test.csv"
    },
    "237_UCR_Anomaly_mit14157longtermecg": {
        "unsupervised": base_data_path / "237_UCR_Anomaly_mit14157longtermecg.test.csv"
    },
    "196_UCR_Anomaly_sel840mECG2": {
        "unsupervised": base_data_path / "196_UCR_Anomaly_sel840mECG2.test.csv"
    },
    "124_UCR_Anomaly_ECG4": {
        "unsupervised": base_data_path / "124_UCR_Anomaly_ECG4.test.csv"
    },
    "238_UCR_Anomaly_mit14157longtermecg": {
        "unsupervised": base_data_path / "238_UCR_Anomaly_mit14157longtermecg.test.csv"
    },
    "121_UCR_Anomaly_ECG3": {
        "unsupervised": base_data_path / "121_UCR_Anomaly_ECG3.test.csv"
    },
    "018_UCR_Anomaly_DISTORTEDECG4": {
        "unsupervised": base_data_path / "018_UCR_Anomaly_DISTORTEDECG4.test.csv"
    },
    "088_UCR_Anomaly_DISTORTEDsel840mECG2": {
        "unsupervised": base_data_path / "088_UCR_Anomaly_DISTORTEDsel840mECG2.test.csv"
    },
    "234_UCR_Anomaly_mit14157longtermecg": {
        "unsupervised": base_data_path / "234_UCR_Anomaly_mit14157longtermecg.test.csv"
    },
    "195_UCR_Anomaly_sel840mECG1": {
        "unsupervised": base_data_path / "195_UCR_Anomaly_sel840mECG1.test.csv"
    },
    "236_UCR_Anomaly_mit14157longtermecg": {
        "unsupervised": base_data_path / "236_UCR_Anomaly_mit14157longtermecg.test.csv"
    },
    "015_UCR_Anomaly_DISTORTEDECG4": {
        "unsupervised": base_data_path / "015_UCR_Anomaly_DISTORTEDECG4.test.csv"
    },
}


class Main:
    _logger = logging.getLogger(__name__)

    @dataclass
    class DatasetSplits(BaseResult):
        train: t.List[str]
        validation: t.List[str]
        test: t.List[str]

        @classmethod
        def load(cls, study_name: str, base_dir: Path) -> "Main.DatasetSplits":
            data = cls._load_dict(
                path=base_dir / study_name / "data_generation" / "dataset_splits.json"
            )
            return cls(
                train=data.get("train"),
                validation=data.get("validation"),
                test=data.get("test"),
            )

        def save(self, study_name: str, base_dir: Path) -> None:
            output_dir = base_dir / study_name / "data_generation"
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_dict(
                path=output_dir / "dataset_splits.json",
                data=self.__dict__,
            )

    def __init__(
        self, cluster_config: hypex.ClusterConfig, seed: int = DEFAULT_SEED
    ) -> None:
        self.cluster_config = cluster_config
        self.seed = seed

    def _compute_with_progress(self, client: dask.distributed.Client, items: t.Any):
        _futures = client.compute(items)
        dask.distributed.progress(_futures)
        return client.gather(_futures)

    def _compute_and_save(
        self,
        client: dask.distributed.Client,
        items: t.Dict[str, t.Any],
        results_dir: Path,
        task_group: str,
    ) -> t.Dict[str, t.Any]:
        self._logger.info(
            "Now submitting the %s task graph to the cluster. The computation will take some time...",
            task_group,
        )
        start = time.time()
        computed_results: t.Dict[str, t.Any] = self._compute_with_progress(
            client,
            items=items,
        )
        end = time.time()
        self._logger.info(
            "The %s computation took %.2f seconds", task_group, end - start
        )

        self._logger.info("Saving %s results to local disk.", task_group)
        for study_name, result in computed_results.items():
            result.save(study_name, results_dir)
        self._logger.info("Completed saving %s results to local disk.", task_group)

        return computed_results

    def _docker_prune(self, cluster: "hypex.ClusterInstance"):
        self._logger.info("Running Docker Prune.")
        _retval = list(cluster.client.run(docker_prune).values())
        self._logger.info("Docker Prune returned %s", _retval)
        self._logger.info("Running Docker Prune Cleanup.")
        _retval = list(cluster.client.run(docker_prune_cleanup).values())
        self._logger.info("Docker Prune Cleanup returned %s", _retval)

    def run(self, config_path: str, results_dir: str = "results") -> None:
        raw_config = hypex.Config.read_yml(config_path)
        studies = hypex.Config.from_dict(config=raw_config)

        results_dir = Path(results_dir)
        self._logger.info("Creating results directory at %s", results_dir.absolute())
        results_dir.mkdir(exist_ok=True)

        existing_studies_without_override = []
        for study in studies:
            if Path(results_dir / study.name).exists():
                self._logger.warn(
                    "The results directory already exists for study %s. [study_override=%s]",
                    study.name,
                    study.study_override,
                )
                if not study.study_override:
                    existing_studies_without_override.append(study.name)
        if existing_studies_without_override:
            raise ValueError(
                "The following studies already have results in the results directory and are not allowed to overwrite: %s",
                existing_studies_without_override,
            )

        score_variable = "auc_pr_score"

        with hypex.Cluster(self.cluster_config) as cluster:
            # Start Redis instance on the scheduler
            cluster.start_optuna_shared_storage()
            cluster.start_optuna_dashboard()
            data_dir = Path("hypex-data")
            get_optuna_storage = self.cluster_config.get_optuna_storage

            generator = hypex.ValueGenerator(seed=self.seed)

            dataset_splits: t.Dict[str, Main.DatasetSplits] = {}
            results_data_generation: t.Dict[str, hypex.DataGenerationModule.Result] = {}
            results_train: t.Dict[str, hypex.CSLModule.Result] = {}

            for study in studies:
                self._logger.info("Starting study %s", study.name)

                # results_data_generation[
                #     study.name
                # ] = hypex.DataGenerationModule.Result(
                #     data_paths={
                #         "synthetic-ts-name-1": data_dir / study.name / "ts-name-1",
                #         "synthetic-ts-name-2": data_dir / study.name / "ts-name-2",
                #         # Real world timeseries
                #         "real-world-ts-name-1": data_dir
                #         / study.name
                #         / "real-world-ts-name-1",
                #     },
                #     applied_mutations={
                #         "ts-name-1": [
                #             {"tsfresh-characteristic-1": "value-1"},
                #             {"tsfresh-characteristic-1": "value-2"},
                #         ],
                #         "ts-name-2": [
                #             {"tsfresh-characteristic-1": "value-1"},
                #             {"tsfresh-characteristic-1": "value-2"},
                #         ],
                #         # Real world timeseries
                #         "real-world-ts-name-1": [
                #             {"tsfresh-characteristic-1": "value-1"},
                #             {"tsfresh-characteristic-1": "value-2"},
                #         ],
                #     },
                #     gutentag_configs={},
                # )
                # dataset_splits[study.name] = Main.DatasetSplits(
                #     train=["ts-name-1", "ts-name-2"],
                #     validation=["ts-name-1", "ts-name-2"],
                #     test=["real-world-ts-name"],
                # )

                if study.start_from_snapshot is not None:
                    name = f"{study.start_from_snapshot.study_name}.{study.timeseries.name}"
                    results_data_generation[
                        study.name
                    ] = hypex.DataGenerationModule.Result.load(
                        study_name=name,
                        base_output_dir=results_dir,
                    )

                    # TODO: remove
                    study_dg_results = results_data_generation[study.name]
                    study_dg_results.data_paths = test_datasets
                    for dataset_name in test_datasets:
                        study_dg_results.applied_mutations[
                            dataset_name
                        ] = get_tsfresh_features(
                            study_dg_results.data_paths[dataset_name]["unsupervised"]
                        )
                    study_dg_results.save(
                        study_name=study.name, base_output_dir=results_dir
                    )
                    dataset_splits[study.name] = Main.DatasetSplits.load(
                        study_name=name,
                        base_dir=results_dir,
                    )
                    # TODO: remove
                    dataset_splits[study.name].test = list(test_datasets.keys())
                    dataset_splits[study.name].save(
                        study_name=study.name, base_dir=results_dir
                    )
                else:
                    results_data_generation[study.name] = hypex.DataGenerationModule(
                        seed=self.seed
                    ).run(
                        cluster=cluster,
                        output_dir=data_dir / study.name,
                        base_data_config=raw_config,
                        base_timeseries_config=study.timeseries,
                        generator=generator,
                    )
                    study_dg_results = results_data_generation[study.name]
                    study_dg_results.data_paths = test_datasets
                    for dataset_name in test_datasets:
                        study_dg_results.applied_mutations[
                            dataset_name
                        ] = get_tsfresh_features(
                            study_dg_results.data_paths[dataset_name]["unsupervised"]
                        )
                    study_dg_results.save(
                        study_name=study.name, base_output_dir=results_dir
                    )

                    (tmp_timeseries_names, timeseries_names_test,) = train_test_split(
                        list(study_dg_results.data_paths.keys()),
                        test_size=DATASET_TEST_SIZE,
                        random_state=self.seed,
                    )
                    (
                        timeseries_names_train,
                        timeseries_names_validation,
                    ) = train_test_split(
                        tmp_timeseries_names,
                        test_size=DATASET_VALIDATION_SIZE,
                        random_state=self.seed,
                    )

                    dataset_splits[study.name] = Main.DatasetSplits(
                        train=timeseries_names_train,
                        validation=timeseries_names_validation,
                        test=list(test_datasets.keys()),
                    )
                    dataset_splits[study.name].save(study.name, results_dir)

                    if len(dataset_splits[study.name].train) == 0:
                        raise ValueError(
                            "No training timeseries found for study %s. Please check the "
                            "configuration file." % study.name
                        )
                    if len(dataset_splits[study.name].validation) == 0:
                        raise ValueError(
                            "No validation timeseries found for study %s. Please check the "
                            "configuration file." % study.name
                        )
                    if len(dataset_splits[study.name].test) == 0:
                        raise ValueError(
                            "No test timeseries found for study %s. Please check the "
                            "configuration file." % study.name
                        )

                # Train
                if (
                    study.start_from_snapshot is not None
                    and study.start_from_snapshot.train
                ):
                    name = f"{study.start_from_snapshot.study_name}.{study.timeseries.name}"
                    results_train[study.name] = hypex.CSLModule.Result.load(
                        study_name=name,
                        base_dir=results_dir,
                    )

                    # results_train[study.name].applied_mutations = {
                    #     name: get_tsfresh_features(paths["unsupervised"])
                    #     for name, paths in results_data_generation[
                    #         study.name
                    #     ].data_paths.items()
                    # }

                    # # TODO: Remove
                    # _results_train = {
                    #     study.name: hypex.CSLModule(seed=self.seed).run(
                    #         study=study,
                    #         trial_results=results_train[study.name].trial_results,
                    #         parameter_importances=results_train[
                    #             study.name
                    #         ].parameter_importances,
                    #         applied_mutations=results_train[
                    #             study.name
                    #         ].applied_mutations,
                    #         score_variable=score_variable,
                    #         parameters=study.parameters,
                    #     )
                    # }
                    # _results_train = self._compute_and_save(
                    #     client=cluster.client,
                    #     items=_results_train,
                    #     results_dir=results_dir,
                    #     task_group="train",
                    # )
                    # results_train.update(_results_train)
                else:
                    trainer = hypex.Trainer(
                        seed=self.seed, get_optuna_storage=get_optuna_storage
                    )
                    optimization_module, _input_data = trainer.prepare_partial(
                        timeseries_names=dataset_splits[study.name].train,
                        result_data_generation=results_data_generation[study.name],
                        study=study,
                    )

                    partial_results = hypex.Trainer.PartialResult.empty()
                    remaining_trials = {
                        ts_name: study.n_trials.train
                        for ts_name in [ele.timeseries_names for ele in _input_data]
                    }
                    while remaining_trials > 0:
                        n_trials = min(TRAIN_BATCH_SIZE, remaining_trials)
                        _partial_results = trainer.run_partial(
                            optimization_module=optimization_module,
                            input_data=_input_data,
                            n_trials=n_trials,
                            trial_group_size=n_trials,
                        )
                        print("_partial_results - 1:", _partial_results)
                        _partial_results = self._compute_with_progress(
                            client=cluster.client,
                            items=_partial_results,
                        )
                        print("_partial_results - 2:", _partial_results)
                        num_successful_trials = len(
                            list(
                                filter(
                                    lambda x: x.exception is None,
                                    _partial_results.trial_results,
                                )
                            )
                        )
                        print("num_successful_trials - 2:", num_successful_trials)
                        partial_results += _partial_results
                        self._docker_prune(cluster=cluster)

                        if num_successful_trials < 0.1 * n_trials:
                            raise RuntimeError(
                                "Less than 10% of the started trials were successful"
                            )

                        remaining_trials -= num_successful_trials

                    _results_train = trainer.finalize_partial(
                        study=study,
                        input_data=_input_data,
                        optimization_module=optimization_module,
                        partial_results=partial_results,
                        result_data_generation=results_data_generation[study.name],
                        timeseries_names=dataset_splits[study.name].train,
                        score_variable=score_variable,
                    )
                    _results_train = self._compute_and_save(
                        client=cluster.client,
                        items=_results_train,
                        results_dir=results_dir,
                        task_group="train",
                    )
                    results_train.update(_results_train)

            # Best Thresholds
            results_best_thresholds: t.Dict[
                str, hypex.Validator.BestThresholdResult
            ] = {}
            studies_to_run_best_threshold: t.List[hypex.Study] = []
            for study in studies:
                if (
                    study.start_from_snapshot is not None
                    and study.start_from_snapshot.best_thresholds
                ):
                    name = f"{study.start_from_snapshot.study_name}.{study.timeseries.name}"
                    results_best_thresholds[
                        study.name
                    ] = hypex.Validator.BestThresholdResult.load(
                        study_name=name,
                        base_dir=results_dir,
                    )
                else:
                    studies_to_run_best_threshold.append(study)

            _results_best_thresholds = hypex.Validator(
                seed=self.seed, get_optuna_storage=get_optuna_storage
            ).run(
                all_timeseries_names={
                    s.name: dataset_splits[s.name].validation
                    for s in studies_to_run_best_threshold
                },
                results_data_generation=results_data_generation,
                results_train=results_train,
                studies=studies_to_run_best_threshold,
                score_variable=score_variable,
            )
            results_best_thresholds.update(_results_best_thresholds)
            results_best_thresholds = self._compute_and_save(
                client=cluster.client,
                items=results_best_thresholds,
                results_dir=results_dir,
                task_group="best-thresholds",
            )

            # Fixed Parameters
            results_fixed_parameters: t.Dict[str, hypex.Validator.Result] = {}
            studies_to_run_fixed_params: t.List[hypex.Study] = []
            for study in studies:
                if (
                    study.start_from_snapshot is not None
                    and study.start_from_snapshot.fixed_parameters
                ):
                    name = f"{study.start_from_snapshot.study_name}.{study.timeseries.name}"
                    results_fixed_parameters[study.name] = hypex.Validator.Result.load(
                        study_name=name,
                        base_dir=results_dir,
                    )
                else:
                    studies_to_run_fixed_params.append(study)

            _results_fixed_parameters = hypex.Validator(
                seed=self.seed, get_optuna_storage=get_optuna_storage
            ).run_determine_fixed_paramters(
                all_timeseries_names={
                    s.name: dataset_splits[s.name].validation
                    for s in studies_to_run_fixed_params
                },
                studies=studies_to_run_fixed_params,
                results_data_generation=results_data_generation,
                results_train=results_train,
                score_variable=score_variable,
                best_threshold_results=results_best_thresholds,
            )
            results_fixed_parameters.update(_results_fixed_parameters)
            results_fixed_parameters = self._compute_and_save(
                client=cluster.client,
                items=results_fixed_parameters,
                results_dir=results_dir,
                task_group="fixed-parameters",
            )

            # Evaluation
            for study in studies:
                results_evaluation: t.Dict[str, hypex.Evaluator.Result] = {}
                if (
                    study.start_from_snapshot is not None
                    and study.start_from_snapshot.test
                ):
                    name = f"{study.start_from_snapshot.study_name}.{study.timeseries.name}"
                    results_evaluation[study.name] = hypex.Evaluator.Result.load(
                        study_name=name,
                        base_dir=results_dir,
                    )
                else:
                    results_evaluation = hypex.Evaluator(
                        seed=self.seed, get_optuna_storage=get_optuna_storage
                    ).run(
                        all_timeseries_names={
                            study.name: dataset_splits[study.name].test
                        },
                        results_data_generation=results_data_generation,
                        results_train=results_train,
                        results_best_thresholds=results_best_thresholds,
                        results_fixed_parameters=results_fixed_parameters,
                        studies=[study],
                    )
                    results_evaluation = self._compute_and_save(
                        client=cluster.client,
                        items=results_evaluation,
                        results_dir=results_dir,
                        task_group="evaluation",
                    )

                self._logger.info("Finished study %s", study.name)
