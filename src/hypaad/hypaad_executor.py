import logging
import typing as t
from pathlib import Path

import dask
import dask.bag
import dask.distributed
import optuna
from gutenTAG import GutenTAG

# pylint: disable=cyclic-import
import hypaad

__all__ = ["HypaadExecutor"]


class HypaadExecutor:
    _logger = logging.getLogger("HypaadExecutor")

    @classmethod
    def _generate_data(
        cls, ts_config: t.Dict[str, t.Any]
    ) -> t.List[t.Dict[str, str]]:
        data_gen = GutenTAG.from_dict(ts_config)
        data_gen.n_jobs = 1
        data_gen.overview.add_seed(1)

        data_gen.generate()

        output_dir = Path("hypaad-data")
        data_gen.save_timeseries(output_dir)

        return [
            {
                ts.dataset_name: {
                    "supervised": output_dir
                    / ts.dataset_name
                    / "train_anomaly.csv",
                    "semi-supervised": output_dir
                    / ts.dataset_name
                    / "train_no_anomaly.csv",
                    "unsupervised": output_dir / ts.dataset_name / "test.csv",
                }
            }
            for ts in data_gen.timeseries
        ]

    @classmethod
    @dask.delayed
    def _create_study(
        cls, study: "hypaad.Study", redis_url: str
    ) -> t.List[str]:
        optuna_study = optuna.create_study(
            study_name=f"Study-{study.name}",
            storage=optuna.storages.RedisStorage(url=redis_url),
            direction="maximize",
        )
        optuna.load_study(
            study_name=optuna_study.study_name,
            storage=optuna.storages.RedisStorage(url=redis_url),
        )
        return [optuna_study.study_name]

    @classmethod
    def _run_study_trial(
        cls,
        trial_id: t.List[int],
        data_paths: t.Dict[str, t.Dict[str, t.Dict[str, Path]]],
        study: "hypaad.Study",
        optuna_study_name: t.List[str],
        redis_url: str,
    ) -> t.List["hypaad.TrialResult"]:
        """Executes a single study trail in the hyper-parameter optimization process.

        Args:
            trial_id: The trial ID.
            data_paths (t.Dict[str, t.Dict[str, t.Dict[str, Path]]]): The paths
                the worker with the given ``executor_id`` stored the generated data at.
            study (hypaad.Study): The study definition.
            optuna_study_name (t.List[str]): Name of the Optuna study.
            redis_url (str): URL of the started Redis instance used to share trial results.

        Raises:
            ValueError: Did not expect more than one executor_id

        Returns:
            t.List[t.Any]:
        """
        if len(trial_id) != 1:
            raise ValueError(
                f"Did not expect more than one trial_id, got {trial_id}"
            )
        worker_address = dask.distributed.get_worker().address
        optimizer = hypaad.Optimizer()
        optimizer.run(
            study=study,
            redis_url=redis_url,
            n_trials=1,
            optuna_study_name=optuna_study_name[0],
            data_paths=data_paths[worker_address][0],
        )
        return optimizer.trial_results

    @classmethod
    def execute(cls, config_path: str):
        """Builds a dask task graph and executes it on a Dask SSHCluster.

        Args:
            config_path (str): Path to the local configuration file
        """
        raw_config = hypaad.Config.read_yml(config_path)
        studies = hypaad.Config.from_dict(config=raw_config)

        with hypaad.Cluster() as cluster:
            data_paths = cluster.client.run(
                cls._generate_data, ts_config=raw_config
            )

            results = {}
            for study in studies:
                redis_url = cluster.get_redis_uri()
                optuna_study_name = cls._create_study(
                    study=study, redis_url=redis_url
                )

                results[study.name] = dask.bag.from_sequence(
                    range(study.ntrials), npartitions=study.ntrials
                ).map_partitions(
                    cls._run_study_trial,
                    data_paths=data_paths,
                    study=study,
                    redis_url=redis_url,
                    optuna_study_name=optuna_study_name,
                )

            cls._logger.info(
                "Now submitting the task graph to the cluster. The computation will take some time..."
            )
            print("result: ", dask.compute(results))
