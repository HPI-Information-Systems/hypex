import json
import logging
import math
import typing as t
from dataclasses import dataclass
from pathlib import Path

import dask
import pandas as pd

import hypaad

from ..base_module import BaseModule
from ..base_result import BaseResult
from .non_linear_pc import NonLinearPC

__all__ = ["CSLModule"]

MIN_ALPHA_THRESHOLD = 0.05
MIN_PARAMETER_IMPORTANCE = 0.01  # 0.05


class CSLModule(BaseModule):

    _logger = logging.getLogger(__name__)

    @dataclass
    class Result(BaseResult):
        csl_candidates: t.Dict[
            t.Tuple[float, float], "hypaad.NonLinearPC.Result"
        ]
        csl_data: pd.DataFrame
        trial_results: pd.DataFrame
        applied_mutations: t.Dict[str, t.Dict[str, str]]
        parameter_importances: t.Dict[str, t.Dict[str, float]]

        @classmethod
        def load(cls, study_name: str, base_dir: Path) -> "CSLModule.Result":
            output_dir = base_dir / study_name / "train"

            applied_mutations = cls._load_dict(
                path=output_dir / "timeseries_mutations.json"
            )

            trial_results = cls._load_dataframe(
                path=output_dir / "trial_results.csv"
            )

            csl_data = cls._load_dataframe(path=output_dir / "csl_data.csv")

            parameter_importances = cls._load_dict(
                path=output_dir / "parameter_importances.json"
            )

            metadata = cls._load_dict(path=output_dir / "metadata.json")

            csl_candidates = {}
            # TODO: Enable
            for item in metadata:
                alpha = item["alpha"]
                beta = item["beta"]
                path_graph_edges = item["path_graph_edges"]
                path_parameter_model = item["path_parameter_model"]

                try:
                    graph_edges = cls._load_dataframe(path=path_graph_edges)
                except pd.errors.EmptyDataError:
                    graph_edges = pd.DataFrame()

                csl_candidates[
                    (float(alpha), float(beta))
                ] = NonLinearPC.Result(
                    parameter_model=hypaad.ParameterModel.load(
                        path_parameter_model,
                    ),
                    graph_edges=graph_edges,
                )

            return cls(
                csl_candidates=csl_candidates,
                csl_data=csl_data,
                trial_results=trial_results,
                applied_mutations=applied_mutations,
                parameter_importances=parameter_importances,
            )

        def save(self, study_name: str, base_output_dir: Path) -> None:
            output_dir = base_output_dir / study_name / "train"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / "timeseries_mutations.json"
            self._save_dict(data=self.applied_mutations, path=output_path)

            output_path = output_dir / "trial_results.csv"
            self._save_dataframe(data=self.trial_results, path=output_path)

            output_path = output_dir / "csl_data.csv"
            self._save_dataframe(data=self.csl_data, path=output_path)

            self._save_dict(
                data=self.parameter_importances,
                path=output_dir / "parameter_importances.json",
            )

            output_dir_graph_edges = output_dir / "csl_graph_edges"
            output_dir_graph_edges.mkdir(parents=True, exist_ok=True)
            output_dir_param_models = output_dir / "parameter_models"
            output_dir_param_models.mkdir(parents=True, exist_ok=True)

            metadata = [
                {
                    "alpha": alpha,
                    "beta": beta,
                    "path_graph_edges": str(
                        output_dir_graph_edges
                        / f"alpha={alpha}_beta={beta}.csv"
                    ),
                    "path_parameter_model": str(
                        output_dir_param_models
                        / f"alpha={alpha}_beta={beta}.pickle"
                    ),
                }
                for alpha, beta in self.csl_candidates.keys()
            ]
            self._save_dict(data=metadata, path=output_dir / "metadata.json")

            for item in metadata:
                alpha_beta = item["alpha"], item["beta"]
                path_graph_edges = item["path_graph_edges"]
                path_parameter_model = item["path_parameter_model"]

                self._save_dataframe(
                    data=self.csl_candidates[alpha_beta].graph_edges,
                    path=path_graph_edges,
                )
                self.csl_candidates[alpha_beta].parameter_model.save(
                    path=path_parameter_model,
                )

    def _prepare(
        self,
        study: "hypaad.Study",
        trial_results: pd.DataFrame,
        parameter_importances: t.Dict[str, t.Dict[str, float]],
        applied_mutations: t.Dict,
        score_variable: str,
    ) -> pd.DataFrame:
        print("parameter_importances: ", parameter_importances)
        mean_parameter_importance = {}
        for importances in parameter_importances.values():
            for parameter, importance in importances.items():
                mean_parameter_importance[parameter] = (
                    mean_parameter_importance.get(parameter, 0) + importance
                )
        mean_parameter_importance = {
            parameter: importance / len(parameter_importances)
            for parameter, importance in mean_parameter_importance.items()
        }
        print('trial_results["params"]', trial_results["params"])
        print('trial_results["params"].loc[0]', trial_results["params"].loc[0])
        print(
            'type(trial_results["params"].loc[0])',
            type(trial_results["params"].loc[0]),
        )
        params_df = pd.json_normalize(
            trial_results["params"].map(
                lambda x: json.loads(x.replace("'", '"'))
                if type(x) == str
                else x
            )
        )
        mutations_df = pd.json_normalize(
            trial_results.timeseries.map(
                lambda x: {
                    entry["name"]: entry["value"]
                    for entry in applied_mutations[x]
                }
            )
        )
        df = pd.concat(
            [params_df, mutations_df],
            axis=1,
        )

        print("mean_parameter_importance: ", mean_parameter_importance)
        print("params_df.columns: ", params_df.columns)
        for col in params_df.columns:
            print(
                f"mean_parameter_importance[{col}]",
                mean_parameter_importance[col],
                mean_parameter_importance[col] >= MIN_PARAMETER_IMPORTANCE,
            )

        pruned_params = [
            col
            for col in params_df.columns.to_list()
            if mean_parameter_importance[col] >= MIN_PARAMETER_IMPORTANCE
        ]
        self._logger.info(
            "Removing columns with low parameter importance. Remaining parameters: %s",
            pruned_params,
        )

        mutation_is_csl_input = {
            m.name: m.is_csl_input for m in study.timeseries.mutations
        }
        included_mutations = [
            name
            for name in mutations_df.columns.to_list()
            if mutation_is_csl_input.get(name, True)
        ]
        self._logger.info(
            "Removing columns representing excluded mutations. Remaining columns: %s [original=%s]",
            included_mutations,
            mutations_df.columns.to_list(),
        )
        df = df[pruned_params + included_mutations]

        df["hypaad_constant"] = 1

        for col in ["timeseries", score_variable]:
            df[col] = trial_results[col]

        return df

    def _apply_filter(
        self,
        data: pd.DataFrame,
        score_variable: str,
        relative_threshold: float,
        cutoff_threshold: float,
    ) -> pd.DataFrame:
        score_cutoff = data[score_variable].max() * cutoff_threshold

        def get_thresh(scores: pd.Series):
            return max(scores.quantile(q=relative_threshold), score_cutoff)

        ts_names = data.timeseries.unique()
        quantiles = {
            ts_name: get_thresh(
                data[data.timeseries == ts_name][score_variable]
            )
            for ts_name in ts_names
        }

        thresholds = pd.DataFrame(
            {
                "timeseries": ts_names,
                "threshold": [quantiles[ts_name] for ts_name in ts_names],
            }
        )

        # thresholds = (
        #     data.groupby("timeseries")
        #     .agg({score_variable: "max"})
        #     .reset_index()
        # )
        # thresholds.rename(columns={score_variable: "score_max"}, inplace=True)
        # theshold_lower_bound = thresholds.score_max.max() * relative_threshold
        # thresholds["threshold"] = theshold_lower_bound

        # (thresholds.score_max * 0.9).apply(
        #     lambda x: max(theshold_lower_bound, x)
        # )

        csl_data = data.copy()
        csl_data["is_csl_input"] = False

        for _, row in thresholds.iterrows():
            ts_name = row["timeseries"]
            threshold = row["threshold"]

            csl_data.loc[
                (csl_data.timeseries == ts_name)
                & (csl_data[score_variable] >= threshold),
                "is_csl_input",
            ] = True

        return csl_data

    def _run(
        self,
        data: pd.DataFrame,
        data_gen_vars: t.List[str],
        score_variable: str,
        parameters: "hypaad.MultidimensionalParameterDistribution",
    ) -> t.Dict[float, NonLinearPC.Result]:
        columns = list(
            set(data.columns.to_list()) - set(["timeseries", "is_csl_input"])
        )
        data_filtered = data[data["is_csl_input"] == True][columns].reset_index(
            drop=True
        )

        self._logger.info("data_filtered.columns: %s", data_filtered.columns)
        self._logger.info("data_gen_vars: %s", data_gen_vars)

        edge_data = NonLinearPC.dry_run(
            data=data_filtered, score_variable=score_variable
        )
        scores = sorted(
            list(
                set(
                    [
                        math.floor(e.score * 100) / 100.0
                        for e in edge_data.values()
                    ]
                )
            ),
            reverse=True,
        )

        # Use scores to determine the threshold
        candidates: t.Dict[t.Tuple[float, float], NonLinearPC.Result] = {}
        for alpha in scores:
            if alpha < MIN_ALPHA_THRESHOLD:
                self._logger.info(
                    "Skipping running CSL with alpha: %f. Lower than minimum threshold of %f",
                    alpha,
                    MIN_ALPHA_THRESHOLD,
                )
                continue
            if alpha == 1.0:
                self._logger.info(
                    "Skipping running CSL with alpha: %f. Equal to 1.0", alpha
                )
                continue

            for beta in [0.2, 0.4, 0.6, 0.8]:
                self._logger.info(
                    "Running CSL with alpha: %f and beta: %f", alpha, beta
                )
                candidates[(alpha, beta)] = NonLinearPC(
                    edge_data=edge_data
                ).run(
                    data=data_filtered,
                    data_gen_vars=data_gen_vars,
                    alpha=alpha,
                    beta=beta,
                    score_variable=score_variable,
                    parameters=parameters,
                )
                self._logger.info(
                    "Finished running CSL with alpha: %f and beta: %f",
                    alpha,
                    beta,
                )

        return candidates

    @dask.delayed
    def run(
        self,
        study: "hypaad.Study",
        trial_results: pd.DataFrame,
        parameter_importances: t.Dict[str, t.Dict[str, float]],
        applied_mutations: t.Dict[str, t.Dict[str, str]],
        score_variable: str,
        parameters: "hypaad.MultidimensionalParameterDistribution",
    ) -> Result:
        distinct_study_names = trial_results["study_name"].unique()
        if len(distinct_study_names) != 1:
            raise ValueError(
                f"Cannot run CSL on multiple studies. Got {distinct_study_names}."
            )

        csl_data = self._prepare(
            study=study,
            trial_results=trial_results,
            parameter_importances=parameter_importances,
            applied_mutations=applied_mutations,
            score_variable=score_variable,
        )

        csl_data = self._apply_filter(
            data=csl_data,
            score_variable=score_variable,
            relative_threshold=study.gamma,
            cutoff_threshold=0.0,
        )

        data_gen_vars = list(
            map(lambda x: x["name"], list(applied_mutations.values())[0])
        ) + ["hypaad_constant"]
        csl_candidates = self._run(
            data=csl_data,
            data_gen_vars=data_gen_vars,
            score_variable=score_variable,
            parameters=parameters,
        )

        return CSLModule.Result(
            csl_data=csl_data,
            csl_candidates=csl_candidates,
            trial_results=trial_results,
            applied_mutations=applied_mutations,
            parameter_importances=parameter_importances,
        )
