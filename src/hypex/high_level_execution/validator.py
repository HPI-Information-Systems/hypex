import json
import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

import dask
import dask.bag
import dask.distributed
import pandas as pd

import hypex

from ..modules.base_result import BaseResult
from .base_runner import BaseRunner

__all__ = ["Validator"]

NUM_RANDOM_PARAM_TRIALS = 10
TRIAL_GROUP_SIZE = 10


class Validator(BaseRunner):
    _logger = logging.getLogger(__name__)

    @dataclass
    class Intermediate:
        alpha: float
        beta: float
        trial_results: pd.DataFrame
        csl_result: "hypex.NonLinearPC.Result"

    @dataclass
    class BestThresholdResult(BaseResult):
        canditate_trial_results: t.Dict[str, pd.DataFrame]
        best_alpha_threshold: float
        best_beta_threshold: float
        mean_scores: t.Dict[str, float]
        median_scores: t.Dict[str, float]
        applied_mutations: t.Dict[str, t.Dict[str, str]]

        @classmethod
        def load(cls, study_name: str, base_dir: Path) -> "Validator.Result":
            output_dir = base_dir / study_name / "best_thresholds"

            metadata = cls._load_dict(path=output_dir / "metadata.json")
            mean_scores = {
                (entry["alpha"], entry["beta"]): entry["value"]
                for entry in metadata["mean_scores"]
            }
            median_scores = {
                (entry["alpha"], entry["beta"]): entry["value"]
                for entry in metadata["median_scores"]
            }

            applied_mutations = cls._load_dict(
                path=output_dir / "timeseries_mutations.json"
            )

            canditate_trial_results = {
                (entry["alpha"], entry["beta"]): cls._load_dataframe(path=entry["path"])
                for entry in metadata["paths"]
            }

            return cls(
                canditate_trial_results=canditate_trial_results,
                best_alpha_threshold=metadata["best_alpha_threshold"],
                best_beta_threshold=metadata["best_beta_threshold"],
                mean_scores=mean_scores,
                median_scores=median_scores,
                applied_mutations=applied_mutations,
            )

        def save(self, study_name: str, base_output_dir: Path) -> None:
            output_dir = base_output_dir / study_name / "best_thresholds"
            output_dir.mkdir(parents=True, exist_ok=True)

            metadata = {
                "best_alpha_threshold": self.best_alpha_threshold,
                "best_beta_threshold": self.best_beta_threshold,
                "mean_scores": [
                    {
                        "alpha": alpha_beta[0],
                        "beta": alpha_beta[1],
                        "value": value,
                    }
                    for alpha_beta, value in self.mean_scores.items()
                ],
                "median_scores": [
                    {
                        "alpha": alpha_beta[0],
                        "beta": alpha_beta[1],
                        "value": value,
                    }
                    for alpha_beta, value in self.median_scores.items()
                ],
                "paths": [
                    {
                        "alpha": alpha_beta[0],
                        "beta": alpha_beta[1],
                        "path": str(
                            output_dir
                            / f"trial_results-alpha={alpha_beta[0]}-beta={alpha_beta[1]}.csv"
                        ),
                    }
                    for alpha_beta in self.canditate_trial_results
                ],
            }
            self._save_dict(data=metadata, path=output_dir / "metadata.json")

            self._save_dict(
                data=self.applied_mutations,
                path=output_dir / "timeseries_mutations.json",
            )

            for entry in metadata["paths"]:
                path = entry["path"]
                alpha_beta = entry["alpha"], entry["beta"]
                self._save_dataframe(
                    data=self.canditate_trial_results[alpha_beta], path=path
                )

    @dataclass
    class Result(BaseResult):
        fixed_parameter_trials: pd.DataFrame
        fixed_parameter_values: t.Dict[str, t.Any]

        @classmethod
        def load(cls, study_name: str, base_dir: Path) -> "Validator.Result":
            output_dir = base_dir / study_name / "fixed_parameters"

            fixed_parameter_values = cls._load_dict(
                path=output_dir / "fixed_parameter_values.json"
            )

            fixed_parameter_trials = cls._load_dataframe(
                path=output_dir / "fixed_parameter_trials.csv"
            )

            return cls(
                fixed_parameter_values=fixed_parameter_values,
                fixed_parameter_trials=fixed_parameter_trials,
            )

        def save(self, study_name: str, base_output_dir: Path) -> None:
            output_dir = base_output_dir / study_name / "fixed_parameters"
            output_dir.mkdir(parents=True, exist_ok=True)

            self._save_dict(
                data=self.fixed_parameter_values,
                path=output_dir / "fixed_parameter_values.json",
            )

            self._save_dataframe(
                data=self.fixed_parameter_trials,
                path=output_dir / "fixed_parameter_trials.csv",
            )

    def _map_parameters(
        self,
        _input: t.List[t.Tuple[str, "hypex.NonLinearPC.Result"]],
        study: "hypex.Study",
        num_random_param_trials: int,
    ):
        if len(_input) != 1:
            raise ValueError(f"Expected one input, got {len(_input)}")
        timeseries_name, csl_result = _input[0]

        registry = hypex.Registry.default()

        model_output_params = csl_result.parameter_model.output_parameters()
        default_params = {
            k: v
            for k, v in registry.get_algorithm(
                name=study.algorithm
            ).default_params.items()
            if k not in model_output_params
        }

        generator = hypex.ValueGenerator(seed=self.seed)
        random_param_sets = [
            {
                k: v
                for k, v in study.parameters.random_guess(generator=generator).items()
                if k not in model_output_params
            }
            for _ in range(num_random_param_trials)
        ]

        return [
            (timeseries_name, csl_result, params, idx)
            for idx, params in enumerate([default_params] + random_param_sets)
        ]

    def _run_algorithm(
        self,
        _input: t.List[
            t.Tuple[
                str,
                "hypex.NonLinearPC.Result",
                t.Dict[str, t.Any],
                int,
            ]
        ],
        study: "hypex.Study",
        result_data_generation: "hypex.DataGenerationModule.Result",
    ) -> t.List[t.Tuple[float, float, "hypex.TrialResult", "hypex.NonLinearPC.Result"]]:
        if len(_input) != 1:
            raise ValueError("Expected one input")
        (
            timeseries_name,
            csl_result,
            predefined_params,
            idx,
        ) = _input[0]

        registry = hypex.Registry.default()

        data_params = {
            entry["name"]: entry["value"]
            for entry in result_data_generation.applied_mutations[timeseries_name]
        }
        data_params["hypex_constant"] = 1

        params = csl_result.parameter_model.predict(**predefined_params, **data_params)
        trial_result = hypex.Optimizer.run_and_score_algorithm(
            trial_id=idx,
            study_name=study.name,
            optuna_study_name="None",
            algorithm=study.algorithm,
            params=params,
            optuna_guess_params={},
            data_paths=result_data_generation.data_paths,
            timeseries_name=timeseries_name,
            registry=registry,
        )

        return [(trial_result, csl_result)]

    def _to_intermediate(
        self,
        _input: t.List[t.Tuple["hypex.TrialResult", "hypex.NonLinearPC.Result"]],
        csl_result_alpha_beta: t.Dict[str, t.List[t.Tuple[float, float]]],
    ):
        trial_results_by_graph_hash = {}
        for entry in _input:
            trial_result, csl_result = entry
            graph_hash = csl_result.get_graph_hash()
            trial_results_by_graph_hash[graph_hash] = trial_results_by_graph_hash.get(
                graph_hash, []
            ) + [trial_result]

        intermediates = []
        for graph_hash, trial_results in trial_results_by_graph_hash.items():
            for alpha, beta in csl_result_alpha_beta[graph_hash]:
                intermediates.append(
                    Validator.Intermediate(
                        alpha=alpha,
                        beta=beta,
                        trial_results=Validator._trials_to_df(trial_results),
                        csl_result=csl_result,
                    )
                )

        return intermediates

    @dask.delayed
    def _get_best_alpha_beta_threshold(
        self,
        intermediates: t.List[Intermediate],
        score_variable: str,
        applied_mutations: t.Dict[str, t.Dict[str, str]],
    ) -> t.Dict[str, BestThresholdResult]:
        canditate_trial_results = {
            (intermediate.alpha, intermediate.beta): intermediate.trial_results
            for intermediate in intermediates
        }
        canditate_csl_results = {
            (intermediate.alpha, intermediate.beta): intermediate.csl_result
            for intermediate in intermediates
        }

        best_alpha_threshold = None
        best_beta_threshold = None
        best_num_edges = 0
        best_score = 0.0
        mean_scores = {}
        median_scores = {}
        for alpha_beta, trial_results in canditate_trial_results.items():
            alpha, beta = alpha_beta
            csl_result = canditate_csl_results[alpha_beta]
            if len(trial_results) == 0:
                self._logger.error(
                    "Did not find any trials for alpha=%f beta=%f. Thus skipping...",
                    alpha,
                    beta,
                )
                continue
            mean_score = trial_results[score_variable].mean()
            mean_scores[alpha_beta] = mean_score
            median_scores[alpha_beta] = trial_results[score_variable].quantile(q=0.5)
            num_edges = len(csl_result.graph_edges)

            mean_score_rounded = round(mean_score, 2)
            best_score_rounded = round(best_score, 2)

            if (
                best_alpha_threshold is None
                or mean_score_rounded > best_score_rounded
                or (
                    mean_score_rounded == best_score_rounded
                    and num_edges < best_num_edges
                )
                or (
                    mean_score_rounded == best_score_rounded
                    and alpha > best_alpha_threshold
                )
            ):
                best_alpha_threshold = alpha
                best_beta_threshold = beta
                best_score = mean_score
                best_num_edges = num_edges

        return Validator.BestThresholdResult(
            canditate_trial_results=canditate_trial_results,
            best_alpha_threshold=best_alpha_threshold,
            best_beta_threshold=best_beta_threshold,
            mean_scores=mean_scores,
            median_scores=median_scores,
            applied_mutations=applied_mutations,
        )

    def determine_fixed_parameters(
        self,
        study: "hypex.Study",
        result_train: "hypex.CSLModule.Result",
        best_threshold_result: BestThresholdResult,
        timeseries_names: t.List[str],
        result_data_generation: "hypex.DataGenerationModule.Result",
        score_variable: str,
    ):
        if best_threshold_result.best_alpha_threshold is not None:
            parameter_model = result_train.csl_candidates[
                (
                    best_threshold_result.best_alpha_threshold,
                    best_threshold_result.best_beta_threshold,
                )
            ].parameter_model
        else:
            parameter_model = None

        fixed_parameter_names = (
            parameter_model.output_parameters()
            if parameter_model is not None
            else set()
        )
        all_parameter_names = set(
            [p.name for p in study.parameters.parameter_distributions]
        )
        remaining_parameter_names = all_parameter_names - fixed_parameter_names

        if len(remaining_parameter_names) == 0:
            return Validator.Result(
                fixed_parameter_values={},
                fixed_parameter_trials=pd.DataFrame(),
            )

        _distribution = hypex.EvaluationModule.get_parameter_distribution_from_training(
            study=study,
            csl_data=result_train.csl_data,
        )

        parameter_distribution = hypex.MultidimensionalParameterDistribution(
            parameter_distributions=[
                d
                for d in _distribution.parameter_distributions
                if d.name in remaining_parameter_names
            ]
        )

        trial_results = (
            hypex.EvaluationModule(seed=self.seed)
            .run(
                get_optuna_storage=self.get_optuna_storage,
                study=study,
                parameter_model=parameter_model,
                timeseries_names=timeseries_names,
                data_paths=result_data_generation.data_paths,
                applied_mutations=best_threshold_result.applied_mutations,
                suffix="validation",
                n_trials=study.n_trials.validation,
                parameter_distribution=parameter_distribution,
                multiple_timeseries_per_study=True,
                trial_group_size=TRIAL_GROUP_SIZE,
            )
            .trial_results
        )
        trial_results = dask.delayed(Validator._trials_to_df)(trial_results)

        @dask.delayed
        def _get_best_params(trial_results: pd.DataFrame):
            print("trial_results.columns: ", trial_results.columns.to_list())
            max_score = trial_results[score_variable].max()
            return trial_results[trial_results[score_variable] == max_score].iloc[0][
                "optuna_guess_params"
            ]

        best_fixed_params = _get_best_params(trial_results)

        return Validator.Result(
            fixed_parameter_values=best_fixed_params,
            fixed_parameter_trials=trial_results,
        )

    def run(
        self,
        all_timeseries_names: t.Dict[str, t.List[str]],
        results_data_generation: t.Dict[str, "hypex.DataGenerationModule.Result"],
        results_train: t.Dict[str, "hypex.CSLModule.Result"],
        studies: t.List[hypex.Study],
        score_variable: str,
    ) -> t.Dict[str, Result]:
        results = {}
        for study in studies:
            result_data_generation = results_data_generation[study.name]
            timeseries_names = all_timeseries_names[study.name]

            applied_mutations = {
                k: v
                for k, v in result_data_generation.applied_mutations.items()
                if k in timeseries_names
            }

            csl_result_alpha_beta = {}
            csl_result_hashes = {}
            for alpha_beta, csl_result in list(
                results_train[study.name].csl_candidates.items()
            ):
                if len(csl_result.graph_edges) > 0:
                    graph_hash = csl_result.get_graph_hash()
                    csl_result_hashes[graph_hash] = csl_result
                    csl_result_alpha_beta[graph_hash] = csl_result_alpha_beta.get(
                        graph_hash, []
                    ) + [alpha_beta]

            _input = [
                (timeseries_name, csl_result_hashes[graph_hash])
                for timeseries_name in timeseries_names
                for graph_hash in csl_result_alpha_beta.keys()
            ]

            results[study.name] = self._get_best_alpha_beta_threshold(
                intermediates=dask.bag.from_sequence(_input, npartitions=len(_input))
                .map_partitions(
                    self._map_parameters,
                    study=study,
                    num_random_param_trials=NUM_RANDOM_PARAM_TRIALS,
                )
                .repartition(npartitions=len(_input) * (1 + NUM_RANDOM_PARAM_TRIALS))
                .map_partitions(
                    self._run_algorithm,
                    study=study,
                    result_data_generation=result_data_generation,
                )
                .repartition(1)
                .map_partitions(
                    self._to_intermediate,
                    csl_result_alpha_beta=csl_result_alpha_beta,
                )
                if len(_input) > 0
                else [],
                score_variable=score_variable,
                applied_mutations=applied_mutations,
            )
        return results

    def run_determine_fixed_paramters(
        self,
        studies: t.List["hypex.Study"],
        all_timeseries_names: t.Dict[str, t.List[str]],
        results_train: t.Dict[str, "hypex.CSLModule.Result"],
        score_variable: str,
        best_threshold_results: t.Dict[str, BestThresholdResult],
        results_data_generation: t.Dict[str, "hypex.DataGenerationModule.Result"],
    ) -> t.Dict[str, "Result"]:
        results = {}
        for study in studies:
            results[study.name] = self.determine_fixed_parameters(
                study=study,
                best_threshold_result=best_threshold_results[study.name],
                timeseries_names=all_timeseries_names[study.name],
                result_train=results_train[study.name],
                result_data_generation=results_data_generation[study.name],
                score_variable=score_variable,
            )
        return results
