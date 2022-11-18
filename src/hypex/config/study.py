import logging
from math import gamma
import typing as t
from dataclasses import dataclass
from uuid import uuid4

import optuna
from torch import seed

import hypex

__all__ = [
    "Study",
    "MultidimensionalParameterDistribution",
    "ParameterDistribution",
    "TimeseriesConfig",
    "TimeseriesMutation",
]

RELATIVE_TRIAL_SCORE_THRESHOLD = 0.9


@dataclass
class TimeseriesMutation:
    name: str
    paths: t.List[str]
    dtype: str
    is_csl_input: bool
    min_value: t.Optional[t.Union[int, float]]
    max_value: t.Optional[t.Union[int, float]]
    choices: t.Optional[t.List[t.Any]]

    def to_kwargs(self) -> t.Dict[str, t.Any]:
        result = {
            "dtype": self.dtype,
        }

        if self.min_value is not None:
            result["min_value"] = self.min_value
        if self.max_value is not None:
            result["max_value"] = self.max_value
        if self.choices is not None:
            result["choices"] = self.choices

        return result

    @classmethod
    def from_config(cls, config: t.Dict[str, t.Any]) -> "TimeseriesMutation":
        return cls(
            name=config.get("name"),
            paths=config.get("paths"),
            dtype=config.get("dtype"),
            is_csl_input=config.get("is_csl_input", True),
            min_value=config.get("min", None),
            max_value=config.get("max", None),
            choices=config.get("choices", None),
        )


@dataclass
class TimeseriesConfig:
    name: str
    n_mutations: int
    mutations: t.List[TimeseriesMutation]

    @classmethod
    def from_config(cls, config: t.Dict[str, t.Any]) -> "TimeseriesConfig":
        return cls(
            name=config.get("name"),
            n_mutations=config.get("n_mutations"),
            mutations=[
                TimeseriesMutation.from_config(mutation)
                for mutation in config.get("mutations")
            ],
        )


@dataclass
class ParameterDistribution:
    name: str
    dtype: str

    # numerical dtype
    min_value: t.Optional[t.Union[int, float]] = None
    max_value: t.Optional[t.Union[int, float]] = None

    # cateogical dtype
    values: t.Optional[t.List[t.Any]] = None

    _logger = logging.getLogger("ParameterDistribution")

    @classmethod
    def from_config(cls, config: t.Dict[str, t.Any]) -> "ParameterDistribution":
        return cls(
            name=config.get("name"),
            dtype=config.get("dtype"),
            min_value=config.get("min", None),
            max_value=config.get("max", None),
            values=config.get("values", None),
        )

    def random_guess(self, generator: "hypex.ValueGenerator") -> t.Any:
        kwargs = {
            "dtype": self.dtype,
        }
        if self.min_value is not None:
            kwargs["min_value"] = self.min_value
        if self.max_value is not None:
            kwargs["max_value"] = self.max_value
        if self.values is not None:
            kwargs["choices"] = self.values

        return generator.run(**kwargs)

    def next_guess(self, trial: optuna.Trial) -> t.Dict[str, t.Any]:
        self._logger.info(
            "Suggesting %s with dtype %s ...", self.name, self.dtype
        )
        if self.dtype == "int":
            return trial.suggest_int(
                name=self.name, low=self.min_value, high=self.max_value
            )
        elif self.dtype == "float":
            return trial.suggest_float(
                name=self.name, low=self.min_value, high=self.max_value
            )
        elif self.dtype == "category":
            return trial.suggest_categorical(
                name=self.name, choices=self.values
            )
        else:
            raise ValueError(f"DType {self.dtype} is currently not supported")


@dataclass
class MultidimensionalParameterDistribution:
    parameter_distributions: t.List["hypex.ParameterDistribution"]

    def next_guess(
        self,
        trial: optuna.Trial,
    ) -> t.Dict[str, t.Any]:
        guess = {}
        for param in self.parameter_distributions:
            if param.name in guess:
                raise ValueError(
                    "Parameter {} is already in the guess".format(param.name)
                )
            guess[param.name] = param.next_guess(trial)
        return guess

    def random_guess(
        self, generator: "hypex.ValueGenerator"
    ) -> t.Dict[str, t.Any]:
        return {
            param.name: param.random_guess(generator=generator)
            for param in self.parameter_distributions
        }

    @classmethod
    def from_config(
        cls, config: t.List[t.Dict[str, t.Any]]
    ) -> "MultidimensionalParameterDistribution":
        return cls(
            parameter_distributions=[
                hypex.ParameterDistribution.from_config(config=param_dist)
                for param_dist in config
            ],
        )


@dataclass
class NumTrials:
    train: int
    validation: int
    test_full_optimization: int
    test_model: int

    @classmethod
    def from_config(
        cls, config: t.Dict[str, int]
    ) -> "MultidimensionalParameterDistribution":
        return cls(
            train=config.get("train"),
            validation=config.get("validation"),
            test_full_optimization=config.get("test_full_optimization"),
            test_model=config.get("test_model"),
        )


@dataclass
class StartFromSnapshot:
    study_name: str
    train: bool
    best_thresholds: bool
    fixed_parameters: bool
    test: bool

    @classmethod
    def from_config(
        cls, config: t.Optional[t.Dict[str, int]]
    ) -> t.Optional["MultidimensionalParameterDistribution"]:
        if config is None:
            return None

        return cls(
            study_name=config.get("study_name"),
            train=config.get("train", False),
            best_thresholds=config.get("best_thresholds", False),
            fixed_parameters=config.get("fixed_parameters", False),
            test=config.get("test", False),
        )


@dataclass
class Study:
    id: str
    name: str
    study_override: bool
    algorithm: str
    n_trials: NumTrials
    timeseries: TimeseriesConfig
    parameters: "hypex.MultidimensionalParameterDistribution"
    start_from_snapshot: t.Optional[StartFromSnapshot]
    gamma: float

    @classmethod
    def studies_from_config(cls, config: t.Dict[str, t.Any]) -> t.List["Study"]:
        studies = []
        study_names = set()
        for timeseries_config in config.get("timeseries"):
            timeseries = TimeseriesConfig.from_config(config=timeseries_config)

            name = f'{config.get("name")}.{timeseries.name}'
            if name in study_names:
                raise ValueError(f"Study name {name} is already in use")
            study_names.add(name)

            studies.append(
                cls(
                    id=str(uuid4()),
                    name=name,
                    study_override=config.get("study_override", False),
                    algorithm=config.get("algorithm"),
                    n_trials=NumTrials.from_config(
                        config=config.get("n_trials")
                    ),
                    timeseries=timeseries,
                    parameters=MultidimensionalParameterDistribution.from_config(
                        config=config.get("parameters")
                    ),
                    start_from_snapshot=StartFromSnapshot.from_config(
                        config=config.get("start_from_snapshot", None)
                    ),
                    gamma=config.get("gamma", RELATIVE_TRIAL_SCORE_THRESHOLD),
                )
            )
        return studies
