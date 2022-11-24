import os
import sys
import typing as t
from dataclasses import dataclass, field

from .optuna_storage import OptunaStorage, OptunaStorageType

__all__ = ["ClusterConfig", "LOCAL_CLUSTER_CONFIG", "REMOTE_CLUSTER_CONFIG"]


@dataclass
# pylint: disable=too-many-instance-attributes
class ClusterConfig:
    scheduler_host: str
    worker_hosts: t.List[str]
    tasks_per_host: int = 1
    task_memory_limit: t.Optional[int] = None
    task_cpu_limit: t.Optional[float] = None

    remote_python: str = sys.executable
    group_privileges: t.Optional[str] = None
    connect_options: t.Dict[str, t.Any] = field(default_factory=dict)

    scheduler_port: int = 8786
    optuna_dashboard_port: int = 8080

    log_level: str = "INFO"
    log_filename: str = "dask.log"

    get_optuna_storage: t.Callable[[], OptunaStorage] = None

    def __post_init__(self):
        if not self.get_optuna_storage:
            port = OptunaStorage.get_port()
            self.get_optuna_storage = lambda: OptunaStorage.get(
                storage_type=OptunaStorageType.MYSQL,
                scheduler_host=self.scheduler_host,
                port=port,
            )

    def dask_scheduler_url(self) -> str:
        return f"{self.scheduler_host}:{self.scheduler_port}"

    def dask_ssh_cluster_config(self) -> t.Dict[str, t.Any]:
        return {
            "hosts": [self.scheduler_host] + self.worker_hosts,
            "connect_options": self.connect_options,
            "worker_options": {
                "nprocs": self.tasks_per_host,
                # "n_workers": self.tasks_per_host,
                "nthreads": 1,
                "memory_limit": self.task_memory_limit or "auto",
            },
            "scheduler_options": {
                "host": self.scheduler_host,
                "port": self.scheduler_port,
            },
            "remote_python": self.remote_python,
        }

    def dask_logging_config(self) -> t.Dict[str, t.Any]:
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "incremental": False,
            "formatters": {
                "brief": {"format": "%(name)s - %(levelname)s - %(message)s"},
                "verbose-file": {
                    "format": "%(asctime)s - %(levelname)s - %(process)d %(name)s - %(message)s"
                },
            },
            "handlers": {
                "stdout": {
                    "level": self.log_level.upper(),
                    "formatter": "brief",
                    "class": "logging.StreamHandler",
                },
                "log_file": {
                    "level": self.log_level.upper(),
                    "formatter": "verbose-file",
                    "filename": self.log_filename,
                    "class": "logging.FileHandler",
                    "mode": "a",
                },
            },
            "root": {"level": "DEBUG", "handlers": ["stdout", "log_file"]},
        }


LOCAL_CLUSTER_CONFIG = ClusterConfig(
    scheduler_host="node-0",
    worker_hosts=["node-0", "node-1"],
    remote_python="python3",
    connect_options={
        "port": 22,
        "username": "admin",
        "password": "secret",
    },
)

# REMOTE_CLUSTER_CONFIG = ClusterConfig(
#     scheduler_host="odin01",
#     worker_hosts=[f"odin{i:02d}" for i in range(1, 15)],
#     remote_python="~/hypex/.venv/bin/python",
# )

SCHEDULER_IP = "172.20.11.101"
WORKER_IPs = [
    "172.20.11.101",
    # "172.20.11.102",  # currently used by other project
    "172.20.11.103",
    "172.20.11.104",
    "172.20.11.105",
    "172.20.11.106",
    "172.20.11.107",
    "172.20.11.108",
    "172.20.11.109",
    "172.20.11.110",
    "172.20.11.111",
    "172.20.11.112",
    "172.20.11.113",
    "172.20.11.114",
]

REMOTE_CLUSTER_CONFIG = ClusterConfig(
    scheduler_host=SCHEDULER_IP,
    worker_hosts=WORKER_IPs,
    remote_python=sys.executable,  # "~/hypex/.venv/bin/python",
    tasks_per_host=os.cpu_count() - 6,
)
