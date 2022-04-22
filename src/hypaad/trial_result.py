import typing as t
from dataclasses import asdict, dataclass

__all__ = ["TrialResult"]


@dataclass
class TrialResult:
    id: int
    algorithm: str
    params: t.Dict[str, t.Any]
    auc_score: float

    def to_dict(self) -> t.Dict[str, t.Any]:
        """Converts the trial result to a dictionary.

        Returns:
            t.Dict[str, t.Any]: The converted trial result.
        """
        res = asdict(self)
        for key, value in self.params.items():
            res[f"params_{key}"] = value
        return res
