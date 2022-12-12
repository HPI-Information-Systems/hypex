#!/usr/bin/env bash
from pathlib import Path

import pandas as pd


def main(folder: Path):
    df = pd.read_csv(folder / "trials.csv")

    results = pd.DataFrame(columns=[
        "study_name", "optuna_study_name", "id", "worker", "algorithm", "timeseries", "optuna_guess_params", "params", "auc_pr_score",
        "roc_auc_score", "best_threshold", "f1_score", "accuracy_score", "anomaly_scores_path", "exception", "is_csl_input",
        "group_items", "params_window_size", "params_n_neighbors", "params_leaf_size", "params_random_state"
    ])
    results["study_name"] = df["study_name"].values
    results["timeseries"] = df["study_name"].str.replace("study-Sub-LOF_real-world-ecg_v2.real-world-ecg-", "", regex=False)\
                                            .str.replace("-no-restrictions", "", regex=False)\
                                            .values
    results["id"] = df["trial_id"].values
    results["auc_pr_score"] = df["ROC_AUC"].values  # I used the wrong name when extracting the target metric
    for param in ["window_size", "n_neighbors", "leaf_size", "random_state"]:
        results[f"params_{param}"] = df[param]
    results["params"] = df[["window_size", "n_neighbors", "leaf_size", "random_state"]].T.apply(dict)
    results["algorithm"] = "Sub-LOF"

    print(results)
    results.to_csv(folder / "trial_results_full_optimization.csv")


if __name__ == '__main__':
    main(Path("results/012-233"))
    main(Path("results/123_UCR_Anomaly_ECG4"))
