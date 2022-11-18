import typing as t
from dataclasses import asdict, dataclass

import hypex

__all__ = ["TrialResult"]


@dataclass
# pylint: disable=too-many-instance-attributes
class TrialResult:
    study_name: str
    optuna_study_name: str
    id: int
    worker: str
    algorithm: str
    timeseries: str
    optuna_guess_params: t.Dict[str, t.Any]
    params: t.Dict[str, t.Any]
    auc_pr_score: float
    roc_auc_score: float
    best_threshold: float
    f1_score: float
    accuracy_score: float
    anomaly_scores_path: str
    exception: t.Optional[Exception] = None
    is_csl_input: t.Optional[bool] = None
    group_items: t.List["hypex.TrialResult"] = None

    def to_dict(self) -> t.Dict[str, t.Any]:
        """Converts the trial result to a dictionary.

        Returns:
            t.Dict[str, t.Any]: The converted trial result.
        """
        res = asdict(self)
        for key, value in self.params.items():
            res[f"params_{key}"] = value
        if self.group_items is not None:
            res["group_items"] = [asdict(item) for item in self.group_items]
        return res
