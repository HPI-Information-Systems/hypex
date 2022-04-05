import typing as t
from dataclasses import dataclass
from uuid import uuid4

import optuna

__all__ = ["Study"]


@dataclass
class Study:
    id: str
    name: str
    algorithm: str
    ntrials: int
    parameters: t.List[t.Any]
    timeseries: t.List[str]

    @classmethod
    def from_config(cls, config: t.Dict[str, t.Any]):
        return cls(
            id=str(uuid4()),
            name=config["name"],
            algorithm=config["algorithm"],
            ntrials=config["ntrials"],
            parameters=config["parameters"],
            timeseries=config["timeseries"],
        )

    def next_parameter_guess(self, trial: optuna.Trial) -> t.Dict[str, t.Any]:
        next_param_guess = {}
        for param in self.parameters:
            name = param["name"]
            dtype = param["dtype"]
            if dtype == "int":
                next_param_guess[name] = trial.suggest_int(
                    name=name, low=param["min"], high=param["max"]
                )
            elif dtype == "float":
                next_param_guess[name] = trial.suggest_float(
                    name=name, low=param["min"], high=param["max"]
                )
            elif dtype == "category":
                next_param_guess[name] = trial.suggest_categorical(
                    name=name, choices=param["values"]
                )
            else:
                raise ValueError(f"DType {dtype} is currently not supported")
        return next_param_guess
