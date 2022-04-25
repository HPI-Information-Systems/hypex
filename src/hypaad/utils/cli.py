import argparse

import hypaad
from hypaad.cluster_config import ClusterConfig
from hypaad.utils.commands import install_dependencies, setup_remote


def main(command: str, environment: str):
    cluster_config: ClusterConfig = None
    if environment == "local":
        cluster_config = hypaad.LOCAL_CLUSTER_CONFIG
    elif environment == "remote":
        cluster_config = hypaad.REMOTE_CLUSTER_CONFIG
    else:
        raise ValueError(f"Unknown environment: {environment}")
    hosts = list(
        set([cluster_config.scheduler_host] + cluster_config.worker_hosts)
    )

    if command == "setup":
        setup_remote(hosts=hosts)
    elif command == "install":
        install_dependencies(hosts=hosts)
    else:
        raise ValueError(f"Unknown command: {command}")


def parse_args():
    parser = argparse.ArgumentParser(description="HYPAAD Utils")
    parser.add_argument(
        "command",
        type=str,
        choices=["setup", "install"],
        help="Command to execute",
    )
    parser.add_argument(
        "-e",
        "--environment",
        type=str,
        required=True,
        choices=["local", "remote"],
        help="Execution environment",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(command=args.command, environment=args.environment)
