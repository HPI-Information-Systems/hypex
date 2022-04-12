import asyncio
import logging
import socket
import sys
import timeit
import typing as t
from time import sleep
from types import TracebackType

import asyncssh
import dask
import dask.distributed
import timeeval
from dask import config as dask_config
from redis import Redis

__all__ = ["Cluster", "ClusterInstance"]

REDIS_PORT = 6379


class ClusterInstance:
    client: t.Optional[dask.distributed.Client] = None
    cluster: t.Optional[dask.distributed.SSHCluster] = None
    redis_container_id: t.Optional[str] = None
    optuna_dashboard_container_id: t.Optional[str] = None

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self.config = timeeval.RemoteConfiguration(
            scheduler_host="node-0",
            worker_hosts=["node-0", "node-1"],
            kwargs_overwrites={
                "connect_options": {
                    "port": 22,
                    "username": "admin",
                    "password": "secret",
                },
                "scheduler_options": {"host": "node-0", "port": 8786},
            },
            remote_python="python3",
        )

        # Setup Dask logging
        self._logger.info("Configuring dask logging")
        dask_config.config["distributed"][
            "logging"
        ] = self.config.get_remote_logging_config()

    def get_client(self) -> dask.distributed.Client:
        return self.client

    def get_workers(self) -> t.List[t.Dict[str, t.Any]]:
        return self.get_client().scheduler_info()["workers"].values()

    def get_num_workers(self) -> int:
        return len(self.get_client().scheduler_info()["workers"])

    async def run_on_worker_ssh(
        self, worker: str, command: str
    ) -> t.Tuple[bool, asyncssh.SSHCompletedProcess]:
        async with asyncssh.connect(
            worker, **self.config.kwargs_overwrites.get("connect_options")
        ) as conn:
            result = await conn.run(command, check=True)
            print(result)
            if result.returncode == 0:
                self._logger.info(
                    "Successfully executed '%s' on host %s", command, worker
                )
                return True, result

            self._logger.error(
                "Failed to execute '%s' on host %s", command, worker
            )
            self._logger.error("Command's error output was %s", result.stderr)
            return False, result

    async def run_on_all_workers_ssh(
        self, command: str
    ) -> t.List[asyncssh.SSHCompletedProcess]:
        failed_on: t.List[str] = []
        results: t.List[asyncssh.SSHCompletedProcess] = []
        for host in self.config.worker_hosts:
            (is_success, result) = await self.run_on_worker_ssh(host, command)
            if not is_success:
                failed_on.append(host)
            results.append(result)
        if len(failed_on) > 0:
            raise RuntimeError(
                f"Execution of '{command}' failed on hosts {failed_on}."
            )
        return results

    def get_redis_uri(self) -> str:
        return f"redis://{self.config.scheduler_host}:{REDIS_PORT}"

    def set_up(self):
        self._logger.info("Creating dask ssh cluster...")
        self.cluster = dask.distributed.SSHCluster(
            **self.config.to_ssh_cluster_kwargs(
                limits=timeeval.ResourceConstraints()
            )
        )
        self.client = dask.distributed.Client(self.cluster.scheduler_address)
        self._logger.info("Successfully connected to the dask cluster")

        self._logger.info("Scheduler Info: %s", self.client.scheduler_info())

        self._logger.info("Setting up Redis container on scheduler...")
        self.redis_container_id = asyncio.run(
            self.run_on_worker_ssh(
                self.config.scheduler_host,
                "docker run -p 0.0.0.0:6379:6379 -d redis:latest",
            )
        )[1].stdout.strip()
        self._logger.info(
            "Redis container started with id %s", self.redis_container_id
        )

        redis = Redis.from_url(self.get_redis_uri(), socket_connect_timeout=1)
        self._wait_until_up_and_running(
            name="Redis",
            ping_func=redis.ping,
        )

        self._logger.info("Now starting Optuna dashboard...")
        self.optuna_dashboard_container_id = asyncio.run(
            self.run_on_worker_ssh(
                self.config.scheduler_host,
                'docker run --net host -d python:3.9 /bin/sh -c "'
                "pip install optuna-dashboard redis"
                f'optuna-dashboard redis://localhost:{REDIS_PORT} --host 0.0.0.0 --port 8080"',
            )
        )[1].stdout.strip()
        self._wait_until_up_and_running(
            name="Optuna Dashboard",
            ping_func=lambda: self._assert_host_port_reachable(
                host=self.config.scheduler_host, port=8080
            ),
        )

    @classmethod
    def _assert_host_port_reachable(cls, host: str, port: int):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host, port))
        try:
            if result != 0:
                raise ValueError(
                    f"Connection to {host}:{port} could not be established"
                )
        finally:
            sock.close()

    def _wait_until_up_and_running(self, name: str, ping_func: t.Callable):
        start = timeit.default_timer()
        self._logger.info("Now waiting for %s to be up and running...", name)
        while True:
            now = timeit.default_timer()
            try:
                ping_func()
                self._logger.info(
                    "%s finally up and running after %i seconds",
                    name,
                    now - start,
                )
                break
            # pylint: disable=broad-except
            except Exception:
                self._logger.info(
                    "%s not up and running after %i seconds", name, now - start
                )
                sleep(5)

    def tear_down(self):
        self._logger.info(
            "Now tearing down all resources",
        )

        if self.optuna_dashboard_container_id:
            self._logger.info(
                "Stopping Optuna Dashboard container %s on scheduler...",
                self.optuna_dashboard_container_id,
            )
            asyncio.run(
                self.run_on_worker_ssh(
                    self.config.scheduler_host,
                    f"docker stop {self.optuna_dashboard_container_id}",
                )
            )
            self._logger.info("Stopped Optuna Dashboard container on scheduler")

        if self.redis_container_id:
            self._logger.info(
                "Stopping Redis container %s on scheduler...",
                self.redis_container_id,
            )
            asyncio.run(
                self.run_on_worker_ssh(
                    self.config.scheduler_host,
                    f"docker stop {self.redis_container_id}",
                )
            )
            self._logger.info("Stopped Redis container on scheduler")

        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()
        self._logger.info(
            "Successfully torn down all resources",
        )


class Cluster:
    instance: t.Optional[ClusterInstance] = None

    def __enter__(self):
        try:
            self.instance = ClusterInstance()
            self.instance.set_up()
            return self.instance
        except Exception as e:
            self.__exit__(*sys.exc_info())
            raise e
        except KeyboardInterrupt as e:
            self.__exit__(*sys.exc_info())
            raise e

    def __exit__(
        self,
        exc_type: t.Optional[t.Type[BaseException]],
        exc: t.Optional[BaseException],
        traceback: t.Optional[TracebackType],
    ):
        if self.instance:
            self.instance.tear_down()
