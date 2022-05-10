import typing as t
from dataclasses import asdict, dataclass

__all__ = ["TrialResult"]


@dataclass
# pylint: disable=too-many-instance-attributes
class TrialResult:
    study_name: str
    id: int
    worker: str
    algorithm: str
    params: t.Dict[str, t.Any]
    auc_pr_score: float
    roc_auc_score: float
    best_threshold: float
    f1_score: float
    accuracy_score: float
    anomaly_scores_path: str
    is_csl_input: t.Optional[bool] = None

    def to_dict(self) -> t.Dict[str, t.Any]:
        """Converts the trial result to a dictionary.

        Returns:
            t.Dict[str, t.Any]: The converted trial result.
        """
        res = asdict(self)
        for key, value in self.params.items():
            res[f"params_{key}"] = value
        return res
