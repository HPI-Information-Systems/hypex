import os
from pathlib import Path

from gutenTAG import GutenTAG

import hypaad


def generate_data(config_path: str, output_dir: str):
    data_generator = GutenTAG.from_yaml(config_path, False, None)
    data_generator.n_jobs = 1
    data_generator.overview.add_seed(1)

    data_generator.generate()
    data_generator.save_timeseries(Path(output_dir))

    output_files = {
        "train_no_anomaly": "train_no_anomaly.csv",
        "train_anomaly": "train_anomaly.csv",
        "test": "test.csv",
    }
    return {
        k: os.path.join(output_dir, file_name)
        for k, file_name in output_files.items()
    }


def main():
    studies = hypaad.Config.load("config.yaml")

    for study in studies:
        optimizer = hypaad.Optimizer(study=study)
        optimizer.run()


if __name__ == "__main__":
    main()
