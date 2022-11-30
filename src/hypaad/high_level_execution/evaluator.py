import json
import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

import dask
import pandas as pd

import hypaad

from ..modules.base_result import BaseResult
from .base_runner import BaseRunner

__all__ = ["Evaluator"]

_logger = logging.getLogger(__name__)


class Evaluator(BaseRunner):
    @dataclass
    class Result(BaseResult):
        trial_results_full_optimization: pd.DataFrame
        trial_results_parameter_model: pd.DataFrame
        trial_results_default_parameters: pd.DataFrame
        trial_results_timeeval_parameters: pd.DataFrame
        applied_mutations: t.Dict[str, t.Dict[str, str]]

        @classmethod
        def load(cls, study_name: str, base_dir: Path) -> "Evaluator.Result":
            output_dir = base_dir / study_name / "evaluation"

            applied_mutations = cls._load_dict(
                path=output_dir / "timeseries_mutations.json"
            )
            trial_results_full_optimization = cls._load_dataframe(
                path=output_dir / "trial_results_full_optimization.csv"
            )
            trial_results_parameter_model = cls._load_dataframe(
                path=output_dir / "trial_results_parameter_model.csv"
            )
            trial_results_default_parameters = cls._load_dataframe(
                path=output_dir / "trial_results_default_parameters.csv"
            )
            trial_results_timeeval_parameters = cls._load_dataframe(
                path=output_dir / "trial_results_timeeval_parameters.csv"
            )

            return cls(
                trial_results_full_optimization=trial_results_full_optimization,
                trial_results_parameter_model=trial_results_parameter_model,
                trial_results_default_parameters=trial_results_default_parameters,
                trial_results_timeeval_parameters=trial_results_timeeval_parameters,
                applied_mutations=applied_mutations,
            )

        def save(self, study_name: str, base_output_dir: Path) -> None:
            output_dir = base_output_dir / study_name / "evaluation"
            output_dir.mkdir(parents=True, exist_ok=True)

            self._save_dict(
                data=self.applied_mutations,
                path=output_dir / "timeseries_mutations.json",
            )

            self._save_dataframe(
                data=self.trial_results_full_optimization,
                path=output_dir / "trial_results_full_optimization.csv",
            )
            self._save_dataframe(
                data=self.trial_results_default_parameters,
                path=output_dir / "trial_results_default_parameters.csv",
            )
            self._save_dataframe(
                data=self.trial_results_parameter_model,
                path=output_dir / "trial_results_parameter_model.csv",
            )
            self._save_dataframe(
                data=self.trial_results_timeeval_parameters,
                path=output_dir / "trial_results_timeeval_parameters.csv",
            )

    def run(
        self,
        all_timeseries_names: t.Dict[str, t.List[str]],
        results_data_generation: t.Dict[str, "hypaad.DataGenerationModule.Result"],
        results_train: t.Dict[str, "hypaad.CSLModule.Result"],
        results_best_thresholds: t.Dict[str, "hypaad.Validator.BestThresholdResult"],
        results_fixed_parameters: t.Dict[str, "hypaad.Validator.Result"],
        studies: t.List[hypaad.Study],
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

            # Full Optimization
            results_full_optimization = hypaad.EvaluationModule(seed=self.seed).run(
                get_optuna_storage=self.get_optuna_storage,
                study=study,
                parameter_model=None,
                timeseries_names=timeseries_names,
                data_paths=result_data_generation.data_paths,
                applied_mutations=applied_mutations,
                suffix="no-restrictions",
                n_trials=study.n_trials.test_full_optimization,
                parameter_distribution=study.parameters,
            )
            trial_results_full_optimization = dask.delayed(Evaluator._trials_to_df)(
                trial_results=results_full_optimization.trial_results
            )

            # Our parameter model
            alpha = results_best_thresholds[study.name].best_alpha_threshold
            beta = results_best_thresholds[study.name].best_beta_threshold
            alpha_beta = alpha, beta

            parameter_model = (
                (results_train[study.name].csl_candidates[alpha_beta].parameter_model)
                if alpha is not None
                else None
            )

            if parameter_model is None:
                _logger.info(
                    "No parameter model found for study %s (alpha=%s, beta=%s)",
                    study.name,
                    alpha,
                    beta,
                )

            result_parameter_model = hypaad.EvaluationModule(
                seed=self.seed
            ).run_with_parameter_model(
                study=study,
                timeseries_names=timeseries_names,
                data_paths=result_data_generation.data_paths,
                applied_mutations=applied_mutations,
                parameter_model=parameter_model,
                fixed_parameters=results_fixed_parameters[
                    study.name
                ].fixed_parameter_values,
            )
            trial_results_parameter_model = dask.delayed(Evaluator._trials_to_df)(
                trial_results=result_parameter_model
            )

            # Default parameters
            result_default_parameters = hypaad.EvaluationModule(
                seed=self.seed
            ).run_with_default_parameters(
                study=study,
                timeseries_names=timeseries_names,
                data_paths=result_data_generation.data_paths,
            )
            trial_results_default_parameters = dask.delayed(Evaluator._trials_to_df)(
                trial_results=result_default_parameters
            )

            # Timeeval parameters
            result_timeeval_parameters = hypex.EvaluationModule(
                seed=self.seed
            ).run_with_timeeval_parameters(
                study=study,
                timeseries_names=timeseries_names,
                data_paths=result_data_generation.data_paths,
                gutentag_configs=result_data_generation.gutentag_configs,
            )
            trial_results_timeeval_parameters = dask.delayed(
                Evaluator._trials_to_df
            )(trial_results=result_timeeval_parameters)

            results[study.name] = Evaluator.Result(
                trial_results_full_optimization=trial_results_full_optimization,
                trial_results_parameter_model=trial_results_parameter_model,
                trial_results_default_parameters=trial_results_default_parameters,
                trial_results_timeeval_parameters=trial_results_timeeval_parameters,
                applied_mutations=applied_mutations,
            )
        return results
