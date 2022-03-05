import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

import optuna
import pandas as pd
from sklearn import metrics

import hypaad

if TYPE_CHECKING:
    from hypaad import Study

__all__ = ["Optimizer"]

REGISTRY = {
    "singular_spectrum_transformation": hypaad.SSTExecutor,
}


class Optimizer:
    _logger = logging.getLogger(__name__)

    def __init__(self, study: "Study"):
        self.study = study

    def _create_objective(self):
        executor_class = REGISTRY.get(self.study.executable)
        data = load_data()
        # X_train_anomaly, Y_train_anomaly = (
        #     data["train_anomaly"][["timestamp", "value-0"]].to_numpy(),
        #     data["train_anomaly"]["is_anomaly"].to_numpy(),
        # )
        # X_train_no_anomaly, Y_train_no_anomaly = (
        #     data["train_no_anomaly"][["timestamp", "value-0"]].to_numpy(),
        #     data["train_no_anomaly"]["is_anomaly"].to_numpy(),
        # )
        X_test, Y_test = (
            data["test"][["timestamp", "value-0"]].to_numpy(),
            data["test"]["is_anomaly"].to_numpy(),
        )

        def func(trial: optuna.Trial):
            params = self.study.next_parameter_guess(trial)
            executor = executor_class(**params)

            # executor.fit(X_train_no_anomaly, Y_train_no_anomaly)
            # executor.fit(X_train_anomaly, Y_train_anomaly)

            pred = executor.predict(X_test)
            precision, recall, _ = metrics.precision_recall_curve(
                y_true=Y_test, probas_pred=pred
            )
            return metrics.auc(recall, precision)

        return func

    def run(self):
        self._logger.info("Starting optimization of study %s", self.study.name)
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        storage = optuna.storages.RDBStorage(url="sqlite:///db.sqlite3")
        optuna_study = optuna.create_study(
            study_name=f"{self.study.name} ({now})",
            storage=storage,
            direction="maximize",
        )
        optuna_study.optimize(
            self._create_objective(),
            n_trials=self.study.ntrials,
            catch=tuple([IndexError]),
        )
        self._logger.info("Study %s completed", self.study.name)
        print(optuna_study.best_trial)


def load_data():
    data_dir = "/Users/mats/git/master-thesis/hypaad/out/demo"
    return {
        filename: pd.read_csv(os.path.join(data_dir, filename, ".csv"))
        for filename in ["test", "train_anomaly", "train_no_anomaly"]
    }
