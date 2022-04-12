import typing as t
from dataclasses import dataclass

__all__ = ["TrialResult"]


@dataclass
class TrialResult:
    id: int
    algorithm: str
    params: t.Dict[str, t.Any]
    auc_score: float
