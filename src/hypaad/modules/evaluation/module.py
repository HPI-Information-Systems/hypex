import logging
import typing as t

import dask
import pandas as pd

import hypaad

from ...registry import Registry
from ..base_module import BaseModule

__all__ = ["EvaluationModule"]


class EvaluationModule(BaseModule):

    _logger = logging.getLogger(__name__)

    def run_with_default_parameters(
        self,
        study: hypaad.Study,
        timeseries_names: t.List[str],
        data_paths: t.Dict[str, str],
        registry=Registry.default(),
    ) -> t.List["hypaad.TrialResult"]:
        default_params = registry.get_algorithm(study.algorithm).default_params
        return [
            dask.delayed(hypaad.Optimizer.run_and_score_algorithm)(
                trial_id=1,
                study_name=study.name,
                optuna_study_name="None",
                algorithm=study.algorithm,
                params=default_params,
                optuna_guess_params={},
                data_paths=data_paths,
                timeseries_name=timeseries_name,
                registry=registry,
            )
            for timeseries_name in timeseries_names
        ]

    def run_with_timeeval_parameters(
        self,
        study: hypaad.Study,
        timeseries_names: t.List[str],
        data_paths: t.Dict[str, str],
        gutentag_configs: t.Dict[str, t.Dict[str, t.Any]],
        registry=Registry.default(),
    ) -> t.List["hypaad.TrialResult"]:
        results = []
        for timeseries_name in timeseries_names:
            timeeval_param_sets = registry.get_algorithm(
                study.algorithm
            ).get_timeeval_params(gutentag_configs.get(timeseries_name))

            for timeeval_params in timeeval_param_sets:
                results.append(
                    dask.delayed(hypaad.Optimizer.run_and_score_algorithm)(
                        trial_id=1,
                        study_name=study.name,
                        optuna_study_name="None",
                        algorithm=study.algorithm,
                        params=timeeval_params,
                        optuna_guess_params={},
                        data_paths=data_paths,
                        timeseries_name=timeseries_name,
                        registry=registry,
                    )
                )
        return results

    def run_with_parameter_model(
        self,
        study: hypaad.Study,
        timeseries_names: t.List[str],
        data_paths: t.Dict[str, str],
        fixed_parameters: t.Dict[str, t.Any],
        applied_mutations: t.Dict[str, t.List[t.Dict[str, t.Any]]],
        parameter_model: t.Optional["hypaad.ParameterModel"] = None,
        registry=Registry.default(),
    ) -> t.List["hypaad.TrialResult"]:
        results = []
        for timeseries_name in timeseries_names:
            data_params = {
                entry["name"]: entry["value"]
                for entry in applied_mutations[timeseries_name]
            }
            data_params["hypaad_constant"] = 1

            our_params = {
                **fixed_parameters,
            }
            if parameter_model is not None:
                our_params.update(
                    parameter_model.predict(**data_params, **fixed_parameters)
                )

            results.append(
                dask.delayed(hypaad.Optimizer.run_and_score_algorithm)(
                    trial_id=1,
                    study_name=study.name,
                    optuna_study_name="None",
                    algorithm=study.algorithm,
                    params=our_params,
                    optuna_guess_params={},
                    data_paths=data_paths,
                    timeseries_name=timeseries_name,
                    registry=registry,
                )
            )
        return results

    @classmethod
    def get_parameter_distribution_from_training(
        cls, study: hypaad.Study, csl_data: pd.DataFrame
    ):
        adjusted_distributions = []
        for original_distribution in study.parameters.parameter_distributions:
            name = original_distribution.name
            dtype = original_distribution.dtype

            if name not in csl_data.columns:
                adjusted_distributions.append(original_distribution)
                continue

            if dtype == "int" or dtype == "float":
                min_value = csl_data[name].min()
                max_value = csl_data[name].max()
                adjusted_distributions.append(
                    hypaad.ParameterDistribution(
                        name=name,
                        dtype=dtype,
                        min_value=int(min_value)
                        if dtype == "int"
                        else float(min_value),
                        max_value=int(max_value)
                        if dtype == "int"
                        else float(max_value),
                    )
                )
            elif dtype == "category":
                adjusted_distributions.append(
                    hypaad.ParameterDistribution(
                        name=name,
                        dtype=dtype,
                        values=list(csl_data[name].unique()),
                    )
                )
            else:
                raise ValueError(f"Unknown dtype {dtype}")

        return hypaad.MultidimensionalParameterDistribution(
            parameter_distributions=adjusted_distributions,
        )

    def run(
        self,
        get_optuna_storage: t.Callable[[], "hypaad.OptunaStorage"],
        study: hypaad.Study,
        timeseries_names: t.List[str],
        data_paths: t.Dict[str, str],
        applied_mutations: t.Dict[str, t.List[t.Dict[str, t.Any]]],
        suffix: str,
        n_trials: int,
        parameter_distribution: hypaad.MultidimensionalParameterDistribution,
        trial_group_size: int = 1,
        multiple_timeseries_per_study: bool = False,
        parameter_model: t.Optional["hypaad.ParameterModel"] = None,
    ) -> "hypaad.OptimizationModule.Result":
        @dask.delayed
        def create_postprocess_parameter_guess_func(
            parameter_model: hypaad.ParameterModel,
        ):
            if parameter_model is None:
                return None

            def func(
                timeseries_name: str,
                **params,
            ) -> t.Callable[[t.Dict[str, t.Any]], t.Dict[str, t.Any]]:
                data_params = {
                    entry["name"]: entry["value"]
                    for entry in applied_mutations[timeseries_name]
                }
                data_params["hypaad_constant"] = 1

                return {
                    **params,
                    **parameter_model.predict(**data_params, **params),
                }

            return func

        postprocess_parameter_guess = create_postprocess_parameter_guess_func(
            parameter_model=parameter_model
        )

        return hypaad.OptimizationModule(
            get_optuna_storage=get_optuna_storage,
            study_name=study.name,
            timeseries_names=timeseries_names,
            data_paths=data_paths,
            algorithm=study.algorithm,
            parameter_distribution=parameter_distribution,
            postprocess_parameter_guess=postprocess_parameter_guess,
            seed=self.seed,
            suffix=suffix,
        ).run(
            n_trials=n_trials,
            trial_group_size=trial_group_size,
            multiple_timeseries_per_study=multiple_timeseries_per_study,
        )
