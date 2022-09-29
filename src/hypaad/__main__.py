import argparse

import hypaad
from hypaad.cluster_config import ClusterConfig


def main(environment: str, config_path: str):
    cluster_config: ClusterConfig = None
    if environment == "local":
        cluster_config = hypaad.LOCAL_CLUSTER_CONFIG
    elif environment == "remote":
        cluster_config = hypaad.REMOTE_CLUSTER_CONFIG
    else:
        raise ValueError(f"Unknown environment: {environment}")

    hypaad.Main(cluster_config=cluster_config).run(config_path=config_path)


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
