import dataclasses
import logging
import random
import typing as t
from pathlib import Path

import dask
import dask.bag
import dask.dataframe
import dask.distributed
import numpy as np
import optuna
import pandas as pd
import portalocker
from gutenTAG import GutenTAG
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# pylint: disable=cyclic-import
import hypaad
from hypaad.csl.pc import run_pc
from hypaad.r_bridge import RBridge

__all__ = ["HypaadExecutor"]

seed = 1


class HypaadExecutor:
    _logger = logging.getLogger("HypaadExecutor")

    @classmethod
    def _generate_data(
        cls, ts_config: t.Dict[str, t.Any]
    ) -> t.List[t.Dict[str, str]]:
        output_dir = Path("hypaad-data")

        try:
            with portalocker.TemporaryFileLock(output_dir / "data-gen.lock"):
                data_gen = GutenTAG.from_dict(ts_config)
                data_gen.n_jobs = 1

                data_gen.overview.add_seed(seed)

                np.random.seed(seed)
                random.seed(seed)

                data_gen.generate()

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
                            "unsupervised": output_dir
                            / ts.dataset_name
                            / "test.csv",
                        }
                    }
                    for ts in data_gen.timeseries
                ]
        except portalocker.AlreadyLocked:
            cls._logger.info(
                "Could not aquire lock. There must be another worker generating the data. "
                "Thus skipping the data generation."
            )
            return []

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
        data_paths: t.Dict[str, t.Dict[str, Path]],
        study: "hypaad.Study",
        optuna_study_name: t.List[str],
        redis_url: str,
    ) -> t.List["hypaad.TrialResult"]:
        """Executes a single study trial in the hyper-parameter optimization process.

        Args:
            trial_id: The trial ID.
            data_paths (t.Dict[str, t.Dict[str, Path]]): The paths
                the generated data was stored at.
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
        optimizer = hypaad.Optimizer()
        optimizer.run(
            study=study,
            redis_url=redis_url,
            n_trials=1,
            optuna_study_name=optuna_study_name[0],
            data_paths=data_paths,
        )
        return optimizer.trial_results

    @classmethod
    @dask.delayed
    def _cluster_trials(
        cls, trial_results: t.List["hypaad.TrialResult"]
    ) -> t.List["hypaad.TrialResult"]:
        X = np.array([r.auc_pr_score for r in trial_results]).reshape(-1, 1)
        best_sil_score, best_cluster_idx, best_labels = 0, 0, []
        for n_clusters in range(2, 5):
            kmeans = KMeans(n_clusters=n_clusters, max_iter=50)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_sil_score:
                cls._logger.info(
                    "Found better trial clustering with silhouette_score=%d n_clusters=%d",
                    score,
                    n_clusters,
                )
                best_sil_score = score
                best_cluster_idx = np.argmax(
                    np.array(kmeans.cluster_centers_).flatten()
                )
                best_labels = np.array(labels).flatten()

        updated_trial_results = []
        for idx, trial_result in enumerate(trial_results):
            _trial_result = dataclasses.replace(trial_result)
            _trial_result.is_csl_input = best_labels[idx] == best_cluster_idx
            updated_trial_results.append(_trial_result)
        return updated_trial_results

    @classmethod
    @dask.delayed
    def _trials_to_df(
        cls, trial_results: t.List["hypaad.TrialResult"]
    ) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            data=[t.to_dict() for t in trial_results],
            orient="columns",
        )

    @classmethod
    @dask.delayed
    def _run_csl(cls, data: pd.DataFrame) -> pd.DataFrame:
        distinct_study_names = data["study_name"].unique()
        if len(distinct_study_names) != 1:
            raise ValueError(
                f"Cannot run CSL on multiple studies. Got {distinct_study_names}."
            )
        cls_input_df = pd.json_normalize(data["params"])
        return run_pc(data=cls_input_df, alpha=0.01)

    @classmethod
    def _compute_with_progress(cls, client, *args, **kwargs):
        future = client.compute(*args, **kwargs)
        dask.distributed.progress(future)
        return client.gather(future)

    @classmethod
    def execute(cls, cluster_config: hypaad.ClusterConfig, config_path: str):
        """Builds a dask task graph and executes it on a Dask SSHCluster.

        Args:
            config_path (str): Path to the local configuration file
        """
        raw_config = hypaad.Config.read_yml(config_path)
        studies = hypaad.Config.from_dict(config=raw_config)

        results_dir = Path("results")
        cls._logger.info(
            "Creating results directory at %s", results_dir.absolute()
        )
        results_dir.mkdir(exist_ok=True)

        with hypaad.Cluster(cluster_config) as cluster:
            _data_paths = list(
                cluster.client.run(
                    cls._generate_data, ts_config=raw_config
                ).values()
            )
            data_paths = {}
            for entry in _data_paths:
                if len(entry) == 1:
                    data_paths.update(entry[0])

            results = {}
            for study in studies:
                redis_url = cluster.get_redis_uri()
                optuna_study_name = cls._create_study(
                    study=study, redis_url=redis_url
                )

                trial_results = (
                    dask.bag.from_sequence(
                        range(study.ntrials), npartitions=study.ntrials
                    )
                    .map_partitions(
                        cls._run_study_trial,
                        data_paths=data_paths,
                        study=study,
                        redis_url=redis_url,
                        optuna_study_name=optuna_study_name,
                    )
                    .repartition(1)
                )

                clustered_trial_results = cls._cluster_trials(
                    trial_results=trial_results
                )
                trial_results_df = cls._trials_to_df(
                    trial_results=clustered_trial_results
                )
                csl_graph = cls._run_csl(data=trial_results_df)

                results[study.name] = {
                    "trial_results": trial_results_df,
                    "csl_graph": csl_graph,
                }

            cls._logger.info(
                "Now submitting the task graph to the cluster. The computation will take some time..."
            )

            computed_results: t.List[
                t.Tuple[str, t.Dict[str, pd.DataFrame]]
            ] = cls._compute_with_progress(cluster.client, results).items()
            for study_name, values in computed_results:
                output_dir = results_dir / study_name
                output_dir.mkdir(exist_ok=True)

                output_path = output_dir / "trial_results.csv"
                cls._logger.info(
                    "Saving trial results of study %s to %s",
                    study_name,
                    output_path,
                )
                values["trial_results"].to_csv(output_path, index=False)

                output_path = output_dir / "csl_graph.csv"
                cls._logger.info(
                    "Saving CSL results of study %s to %s",
                    study_name,
                    output_path,
                )
                RBridge.transform_to_pandas_dataframe(
                    values["csl_graph"]
                ).to_csv(output_path, index=False)

            cls._logger.info("Completed saving study results to local disk.")
