import asyncio
import logging
import socket
import sys
import timeit
import typing as t
from glob import glob
from time import sleep
from types import TracebackType

import asyncssh
import dask
import dask.distributed
import numpy as np
from dask import config as dask_config

# pylint: disable=cyclic-import,unused-import
import hypaad
from hypaad.optuna_storage import OptunaStorage

__all__ = ["Cluster", "ClusterInstance"]


class ClusterInstance:
    client: t.Optional[dask.distributed.Client] = None
    cluster: t.Optional[dask.distributed.SSHCluster] = None
    running_container_ids: t.List[str] = []
    optuna_dashboard_container_id: t.Optional[str] = None

    def __init__(self, config: "hypaad.ClusterConfig") -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        # Setup Dask logging
        self._logger.info("Configuring dask logging")
        dask_config.config["distributed"]["logging"] = self.config.dask_logging_config()

    def get_client(self) -> dask.distributed.Client:
        return self.client

    def get_workers(self) -> t.List[t.Dict[str, t.Any]]:
        return self.get_client().scheduler_info()["workers"].values()

    def get_num_workers(self) -> int:
        return len(self.get_client().scheduler_info()["workers"])

    async def run_on_worker_ssh(
        self, worker: str, command: str, timeout: t.Optional[float] = None
    ) -> t.Tuple[bool, asyncssh.SSHCompletedProcess]:
        """Runs a ``command`` on a given ``worker`` node via SSH.

        Args:
            worker (str): The node to execute the command on.
            command (str): The command to execute on the ``worker``.
            timeout (t.Optional[float], optional): A number of seconds to wait
                before considering the command as failed. Defaults to None.

        Raises:
            RuntimeError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            t.Tuple[bool, asyncssh.SSHCompletedProcess]: _description_
        """
        async with asyncssh.connect(worker, **self.config.connect_options) as conn:
            result = await conn.run(command, timeout=timeout, check=True)
            if result.returncode == 0:
                self._logger.info(
                    "Successfully executed '%s' on host %s", command, worker
                )
                return True, result

            self._logger.error("Failed to execute '%s' on host %s", command, worker)
            self._logger.error("Command's error output was %s", result.stderr)
            return False, result

    async def run_on_all_workers_ssh(
        self, command: str, timeout: t.Optional[float] = None
    ) -> t.List[asyncssh.SSHCompletedProcess]:
        failed_on: t.List[str] = []
        results: t.List[asyncssh.SSHCompletedProcess] = []
        for host in self.config.worker_hosts:
            (is_success, result) = await self.run_on_worker_ssh(
                worker=host, command=command, timeout=timeout
            )
            if not is_success:
                failed_on.append(host)
            results.append(result)
        if len(failed_on) > 0:
            raise RuntimeError(f"Execution of '{command}' failed on hosts {failed_on}.")
        return results

    def set_up(self):
        try:
            c = dask.distributed.Client(self.config.dask_scheduler_url(), timeout=1)
            self._logger.warning(
                "Found a running scheduler. Thus shutting it down now..."
            )
            c.shutdown()
        # Connection timeout
        except OSError:
            self._logger.info("No running scheduler found.")

        self._logger.info("Creating dask SSH cluster...")
        self.cluster = dask.distributed.SSHCluster(
            **self.config.dask_ssh_cluster_config()
        )
        self.client = dask.distributed.Client(self.cluster.scheduler_address)
        self._logger.info("Successfully connected to the dask cluster")

        self._logger.info("Scheduler Info: %s", self.client.scheduler_info())

        self._logger.info(
            "Now waiting for %d workers to be up and running...",
            len(self.config.worker_hosts) * self.config.tasks_per_host,
        )
        self.client.wait_for_workers(len(self.config.worker_hosts))
        self._logger.info(
            "All %d workers are now up and running. Proceding setup",
            len(self.config.worker_hosts) * self.config.tasks_per_host,
        )

        self._logger.info("Uploading local changes to cluster...")
        self._upload_local_code_changes()
        self._logger.info("Sucessfully uploaded local changes to cluster")

    def _upload_local_code_changes(self):
        FILTER = "./dist/*.egg"
        dist_files = glob(FILTER)
        if len(dist_files) == 0:
            raise ValueError(
                f"Expected to find a packaged version of hypex with filter '{FILTER}'"
            )
        if len(dist_files) > 1:
            raise ValueError(
                f"Did not expect to find more than one file with filter '{FILTER}'. Found {dist_files}."
            )
        self.client.upload_file(dist_files[0])

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

    def start_optuna_shared_storage(self) -> str:
        storage = self.config.get_optuna_storage()
        self._logger.info(
            "Setting up %s container on scheduler...",
            storage.type,
        )
        container_id = asyncio.run(
            self.run_on_worker_ssh(
                worker=self.config.scheduler_host,
                command=storage.get_docker_command(),
                timeout=60,
            )
        )[1].stdout.strip()
        self._logger.info(
            "%s container started with id %s",
            storage.type,
            container_id,
        )
        self.running_container_ids.append(container_id)
        url = storage.get_uri()

        self._wait_until_up_and_running(
            name=storage.type,
            ping_func=storage.get_ping_func(),
        )
        return url

    def start_optuna_dashboard(self, wait=False):
        self._logger.info("Now starting Optuna dashboard...")
        storage_url = self.config.get_optuna_storage().get_uri()
        self.optuna_dashboard_container_id = asyncio.run(
            self.run_on_worker_ssh(
                worker=self.config.scheduler_host,
                command=f"docker run --net host "
                '-d python:3.9 /bin/sh -c "pip install optuna-dashboard redis mysqlclient '
                f"&& optuna-dashboard {storage_url} "
                f'--host 0.0.0.0 --port {self.config.optuna_dashboard_port}"',
                timeout=60,
            )
        )[1].stdout.strip()

        if wait:
            self._wait_until_up_and_running(
                name="Optuna Dashboard",
                ping_func=lambda: self._assert_host_port_reachable(
                    host=self.config.scheduler_host,
                    port=self.config.optuna_dashboard_port,
                ),
                timeout=60,
            )

    def _wait_until_up_and_running(
        self,
        name: str,
        ping_func: t.Callable,
        timeout: t.Optional[float] = None,
    ):
        start = timeit.default_timer()
        self._logger.info("Now waiting for %s to be up and running...", name)
        while True:
            now = timeit.default_timer()
            if timeout and now - start >= timeout:
                msg = f"{name} still not up and running after {now - start} seconds, but timeout was {timeout} seconds."
                self._logger.error(msg)
                raise TimeoutError(msg)
            try:
                ping_func()
                self._logger.info(
                    "%s finally up and running after %i seconds",
                    name,
                    now - start,
                )
                break
            # pylint: disable=broad-except
            except Exception as e:
                suffix = f" [timeout={timeout}]" if timeout else ""
                self._logger.info(
                    "%s not up and running after %i seconds%s [Exception=%s]",
                    name,
                    now - start,
                    suffix,
                    e,
                )
                sleep(5)

    def tear_down(self, reason: str):
        self._logger.info("Now tearing down all resources. Reason: %s", reason)

        if self.optuna_dashboard_container_id:
            self._logger.info(
                "Stopping Optuna Dashboard container %s on scheduler...",
                self.optuna_dashboard_container_id,
            )
            asyncio.run(
                self.run_on_worker_ssh(
                    worker=self.config.scheduler_host,
                    command=f"docker stop {self.optuna_dashboard_container_id} "
                    f"&& docker container rm {self.optuna_dashboard_container_id}",
                )
            )
            self._logger.info("Stopped Optuna Dashboard container on scheduler")

        storage = self.config.get_optuna_storage()
        for idx, container_id in enumerate(self.running_container_ids):
            self._logger.info(
                "Stopping %s container %s on scheduler [%d/%d] ...",
                storage.type,
                container_id,
                idx + 1,
                len(self.running_container_ids),
            )
            asyncio.run(
                self.run_on_worker_ssh(
                    worker=self.config.scheduler_host,
                    command=f"docker stop {container_id}"
                    f"&& docker container rm {container_id}",
                )
            )
            self._logger.info(
                "Stopped %s container %s on scheduler [%d/%d]",
                storage.type,
                container_id,
                idx + 1,
                len(self.running_container_ids),
            )

        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()
        self._logger.info(
            "Successfully torn down all resources",
        )


class Cluster:
    instance: t.Optional[ClusterInstance] = None

    def __init__(self, config: "hypaad.ClusterConfig"):
        self.config = config

    def __enter__(self):
        try:
            self.instance = ClusterInstance(config=self.config)
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
            if exc_type is not None and exc_type == KeyboardInterrupt:
                reason = "HYPEX was stopped by user."
            elif exc is not None:
                reason = f"HYPEX ran into an error ({exc})"
            else:
                reason = "HYPEX finished processing all trails."
            self.instance.tear_down(reason)
