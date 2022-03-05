import typing as t
from dataclasses import dataclass

import optuna

__all__ = ["Study"]


@dataclass
class Study:
    name: str
    executable: str
    ntrials: int
    parameters: t.List[t.Any]

    @classmethod
    def from_config(cls, config: t.Dict[str, t.Any]):
        return cls(
            name=config["name"],
            executable=config["executable"],
            ntrials=config["ntrials"],
            parameters=config["parameters"],
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
            else:
                raise ValueError(f"DType {dtype} is currently not supported")
        return next_param_guess
