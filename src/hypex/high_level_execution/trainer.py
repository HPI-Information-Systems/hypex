import typing as t
from dataclasses import dataclass

import dask

import hypex

from .base_runner import BaseRunner

__all__ = ["Trainer"]


class Trainer(BaseRunner):
    @dataclass
    class PartialResult:
        trial_results: t.List["hypex.TrialResult"]

        def __add__(self, other: "Trainer.PartialResult"):
            if not isinstance(other, self.__class__):
                raise ValueError(
                    f"Cannot add {other.__class__} to PartialResult"
                )
            trial_results = self.trial_results
            trial_results.extend(other.trial_results)
            return Trainer.PartialResult(trial_results=trial_results)

        @classmethod
        def empty(cls):
            return cls(trial_results=[])

    def prepare_partial(
        self,
        timeseries_names: t.List[str],
        result_data_generation: "hypex.DataGenerationModule.Result",
        study: hypex.Study,
    ) -> t.Tuple[
        "hypex.OptimizationModule",
        t.List["hypex.OptimizationModule.Intermediate"],
    ]:
        optimization_module = hypex.OptimizationModule(
            storage=self.storage,
            study_name=study.name,
            timeseries_names=timeseries_names,
            data_paths=result_data_generation.data_paths,
            algorithm=study.algorithm,
            parameter_distribution=study.parameters,
            seed=self.seed,
            suffix="train",
        )
        return optimization_module, optimization_module.create_studies()

    def run_partial(
        self,
        optimization_module: "hypex.OptimizationModule",
        input_data: t.List["hypex.OptimizationModule.Intermediate"],
        n_trials: int,
        trial_group_size: int,
    ) -> "Trainer.PartialResult":
        partial_trial_result = optimization_module.run_partial(
            input_data=input_data,
            n_trials=n_trials,
            trial_group_size=trial_group_size,
        )
        return Trainer.PartialResult(
            trial_results=partial_trial_result,
        )

    def finalize_partial(
        self,
        study: "hypex.Study",
        input_data: t.List["hypex.OptimizationModule.Intermediate"],
        optimization_module: "hypex.OptimizationModule",
        partial_results: "PartialResult",
        result_data_generation: "hypex.DataGenerationModule.Result",
        timeseries_names: t.List[str],
        score_variable: str,
    ):
        optimization_result = optimization_module.finalize_partial(
            input_data=input_data,
            trial_results=partial_results.trial_results,
        )
        trial_results = dask.delayed(Trainer._trials_to_df)(
            trial_results=partial_results.trial_results,
        )

        applied_mutations = {
            k: v
            for k, v in result_data_generation.applied_mutations.items()
            if k in timeseries_names
        }

        return {
            study.name: hypex.CSLModule(seed=self.seed).run(
                study=study,
                trial_results=trial_results,
                parameter_importances=optimization_result.parameter_importances,
                applied_mutations=applied_mutations,
                score_variable=score_variable,
                parameters=study.parameters,
            )
        }

    def run(
        self,
        all_timeseries_names: t.Dict[str, t.List[str]],
        results_data_generation: t.Dict[
            str, "hypex.DataGenerationModule.Result"
        ],
        study: hypex.Study,
        score_variable: str,
    ) -> t.Dict[str, "hypex.CSLModule.Result"]:
        result_data_generation = results_data_generation[study.name]
        timeseries_names = all_timeseries_names[study.name]

        applied_mutations = {
            k: v
            for k, v in result_data_generation.applied_mutations.items()
            if k in timeseries_names
        }

        optimiztion_result = hypex.OptimizationModule(
            storage=self.storage,
            study_name=study.name,
            timeseries_names=timeseries_names,
            data_paths=result_data_generation.data_paths,
            algorithm=study.algorithm,
            parameter_distribution=study.parameters,
            seed=self.seed,
            suffix="train",
        ).run(n_trials=study.n_trials.train)

        trial_results = dask.delayed(Trainer._trials_to_df)(
            trial_results=optimiztion_result.trial_results,
        )

        return {
            study.name: hypex.CSLModule(seed=self.seed).run(
                study=study,
                trial_results=trial_results,
                parameter_importances=optimiztion_result.parameter_importances,
                applied_mutations=applied_mutations,
                score_variable=score_variable,
                parameters=study.parameters,
            )
        }
