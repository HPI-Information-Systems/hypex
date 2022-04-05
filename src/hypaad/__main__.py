import argparse

import hypaad


def main(environment: str, config_path: str):
    if environment != "local":
        raise ValueError("Currently only local execution is supported")
    hypaad.HypaadExecutor.execute(config_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="HYPAAD - Hyper Parameter Optimization in Time Series Anomaly Detection"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "-e",
        "--environment",
        type=str,
        default="local",
        choices=["local", "remote"],
        help="Execution environment",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(environment=args.environment, config_path=args.config)
