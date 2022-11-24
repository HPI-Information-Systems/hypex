import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

import dask.bag
import optuna

import hypaad
from hypaad.optuna_storage import OptunaStorage

from ..base_module import BaseModule
from .optimizer import Optimizer

__all__ = ["OptimizationModule"]


class OptimizationModule(BaseModule):
    _logger = logging.getLogger(__name__)

    @dataclass
    class Intermediate:
        study_name: str
        optuna_study_name: str
        timeseries_names: t.List[str]

    @dataclass
    class Result:
        trial_results: t.List["hypaad.TrialResult"]
        parameter_importances: t.Dict[str, t.Dict[str, float]]

    def __init__(
        self,
        algorithm: str,
        get_optuna_storage: t.Callable[[], OptunaStorage],
        study_name: str,
        timeseries_names: str,
        data_paths: t.Dict[str, t.Dict[str, Path]],
        parameter_distribution: hypaad.MultidimensionalParameterDistribution,
        seed: int,
        suffix: str,
        n_startup_trials: int = 20,
        postprocess_parameter_guess: t.Optional[t.Callable] = None,
        registry: t.Optional["hypaad.Registry"] = None,
    ) -> None:
        super().__init__(seed)
        self.algorithm = algorithm
        self.get_optuna_storage = get_optuna_storage
        self.study_name = study_name
        self.timeseries_names = timeseries_names
        self.data_paths = data_paths
        self.parameter_distribution = parameter_distribution
        self.postprocess_parameter_guess = postprocess_parameter_guess
        self.n_startup_trials = n_startup_trials
        self.registry = registry
        self.suffix = suffix

    @classmethod
    def _create_study(
        cls,
        timeseries_name: str,
        study_name: str,
        n_startup_trials: str,
        seed: int,
        suffix: str,
        get_optuna_storage: t.Callable[[], "hypaad.OptunaStorage"],
    ) -> Intermediate:
        storage = get_optuna_storage()
        optuna_study = optuna.create_study(
            study_name=f"study-{study_name}-{timeseries_name}-{suffix}",
            storage=storage.get_storage_backend(),
            sampler=optuna.samplers.TPESampler(
                multivariate=False,
                n_ei_candidates=15,
                constant_liar=True,
                n_startup_trials=n_startup_trials,
                warn_independent_sampling=True,
                seed=seed,
            ),
            # sampler=optuna.samplers.CmaEsSampler(
            #     n_startup_trials=n_startup_trials,
            #     warn_independent_sampling=True,
            #     seed=seed,
            # ),
            direction="maximize",
        )
        cls._logger.info(
            "Successfully created the study %s with ID=%d",
            optuna_study.study_name,
            optuna_study._study_id,
        )
        optuna.load_study(
            study_name=optuna_study.study_name,
            storage=storage.get_storage_backend(),
        )
        return cls.Intermediate(
            study_name=study_name,
            optuna_study_name=optuna_study.study_name,
            timeseries_names=[timeseries_name],
        )

    @classmethod
    @dask.delayed
    def _get_result(
        cls,
        intermediates: t.List[Intermediate],
        get_optuna_storage: t.Callable[[], "hypaad.OptunaStorage"],
        trial_results: t.List["hypaad.TrialResult"],
    ) -> Result:
        importances = {}
        for intermediate in intermediates:
            study = optuna.load_study(
                study_name=intermediate.optuna_study_name,
                storage=get_optuna_storage().get_storage_backend(),
            )

            importance = {}
            try:
                importance = optuna.importance.get_param_importances(study=study)
            except Exception as e:
                cls._logger.error(
                    "Failed to get parameter importances for study %s: %s",
                    intermediate.study_name,
                    e,
                )

            importances[intermediate.study_name] = importance
        return OptimizationModule.Result(
            trial_results=trial_results, parameter_importances=importances
        )

    @classmethod
    def _run_trial(
        cls,
        intermediates: t.List[Intermediate],
        batch_size: int,
        get_optuna_storage: t.Callable[[], "hypaad.OptunaStorage"],
        parameter_distribution: "hypaad.MultidimensionalParameterDistribution",
        algorithm: str,
        study_name: str,
        data_paths: t.Dict[str, t.Dict[str, Path]],
        registry: "hypaad.Registry",
        postprocess_parameter_guess: t.Optional[t.Callable] = None,
        **kwargs: t.Any,
    ) -> t.List["hypaad.TrialResult"]:
        if len(intermediates) != 1:
            raise ValueError("intermediates must contain a single element.")

        intermediate = intermediates[0]

        # if (
        #     postprocess_parameter_guess is not None
        #     and len(intermediate.timeseries_names) != 1
        # ):
        #     raise ValueError(
        #         "postprocess_parameter_guess can only be used with a single timeseries."
        #     )

        # postprocess_parameter_guess = (
        #     get_postprocess_parameter_guess(
        #         timeseries_name=intermediate.timeseries_names[0],
        #     )
        #     if get_postprocess_parameter_guess is not None
        #     else None
        # )

        if len(parameter_distribution.parameter_distributions) == 0:
            return []

        return Optimizer(
            storage=get_optuna_storage(),
            parameter_distribution=parameter_distribution,
            postprocess_parameter_guess=postprocess_parameter_guess,
            timeseries_names=intermediate.timeseries_names,
            algorithm=algorithm,
            data_paths=data_paths,
            registry=registry,
            study_name=study_name,
            optuna_study_name=intermediate.optuna_study_name,
        ).run(
            n_trials=batch_size,
        )

    def create_studies(self):
        return [
            OptimizationModule._create_study(
                timeseries_name=ts_name,
                study_name=self.study_name,
                n_startup_trials=self.n_startup_trials,
                get_optuna_storage=self.get_optuna_storage,
                seed=self.seed,
                suffix=self.suffix,
            )
            for ts_name in self.timeseries_names
        ]

    def create_multi_timeseries_study(self) -> Intermediate:
        _retval = OptimizationModule._create_study(
            timeseries_name="all-timeseries",
            study_name=self.study_name,
            n_startup_trials=self.n_startup_trials,
            get_optuna_storage=self.get_optuna_storage,
            seed=self.seed,
            suffix=self.suffix,
        )
        return [
            OptimizationModule.Intermediate(
                study_name=_retval.study_name,
                optuna_study_name=_retval.optuna_study_name,
                timeseries_names=self.timeseries_names,
            )
        ]

    def run_partial(
        self,
        input_data: t.List[Intermediate],
        n_trials: int,
        trial_group_size: int,
    ) -> t.List["hypaad.TrialResult"]:
        retval = []
        for idx in range(0, n_trials, trial_group_size):
            n_trials_in_group = min(trial_group_size, n_trials - idx)
            retval.append(
                dask.bag.from_sequence(
                    input_data * n_trials_in_group,
                    npartitions=len(input_data) * n_trials_in_group,
                ).map_partitions(
                    OptimizationModule._run_trial,
                    batch_size=1,
                    study_name=self.study_name,
                    get_optuna_storage=self.get_optuna_storage,
                    parameter_distribution=self.parameter_distribution,
                    algorithm=self.algorithm,
                    data_paths=self.data_paths,
                    registry=self.registry,
                    postprocess_parameter_guess=self.postprocess_parameter_guess,
                    previous_group=retval[-1] if idx > 0 else None,
                )
            )

        return dask.bag.concat(retval)

    def finalize_partial(
        self,
        input_data: t.List[Intermediate],
        trial_results: t.List["hypaad.TrialResult"],
    ) -> Result:
        return OptimizationModule._get_result(
            intermediates=input_data,
            trial_results=trial_results,
            get_optuna_storage=self.get_optuna_storage,
        )

    def run(
        self,
        n_trials: int,
        trial_group_size: int,
        multiple_timeseries_per_study: bool = False,
    ) -> Result:
        input_data = (
            self.create_studies()
            if not multiple_timeseries_per_study
            else self.create_multi_timeseries_study()
        )
        trial_results = self.run_partial(
            input_data=input_data,
            n_trials=n_trials,
            trial_group_size=trial_group_size,
        )
        return self.finalize_partial(
            input_data=input_data,
            trial_results=trial_results,
        )
