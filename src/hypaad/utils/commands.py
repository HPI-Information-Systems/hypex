import logging
import os
import shutil
import typing as t
from pathlib import Path

from pssh.clients import ParallelSSHClient

__all__ = ["setup_remote", "install_dependencies", "run_command"]

_logger = logging.getLogger(__name__)


def _upload_file(
    client: ParallelSSHClient,
    path: str,
    temp_path: str,
    dest_path: str,
    callback: t.Callable,
):
    # Make a local copy from the file to upload
    shutil.copyfile(path, temp_path)

    src = Path(temp_path).absolute()
    _logger.info("Now copying file from %s to %s on remote hosts", src, dest_path)
    client.copy_file(src, dest_path)
    client.join()
    os.remove(src)

    callback()


def _run_command(client: ParallelSSHClient, command: str):
    output = client.run_command(command)
    for host_output in output:
        for line in host_output.stdout:
            _logger.info(line)
        _logger.info("Exit code: %d", host_output.exit_code)


def _run_command_callback(client: ParallelSSHClient, cmd: str):
    def func():
        _logger.info("Now running callback on remote hosts")
        _run_command(
            client=client,
            command=cmd,
        )

    return func


def run_command(hosts: t.List[str], command: str):
    client = ParallelSSHClient(hosts)
    _run_command(
        client=client,
        command=command,
    )


def setup_remote(hosts: t.List[str]):
    client = ParallelSSHClient(hosts, pkey="~/.ssh/id_rsa")

    _run_command(
        client=client,
        command="mkdir -p ~/hypex",
    )

    _upload_file(
        client=client,
        path="system-setup.sh",
        temp_path="system-setup.tmp.sh",
        dest_path="~/hypex/system-setup.sh",
        callback=_run_command_callback(client, cmd="./hypex/system-setup.sh"),
    )

    _run_command(
        client=client,
        command="python3 -m venv ~/hypex/.venv && mkdir -p ~/trial_results",
    )
    _install_dependencies(client=client)


def _install_dependencies(client: ParallelSSHClient):
    _logger.info("Now adding gitlab.hpi.de to known hosts")
    _run_command(
        client=client, command="ssh-keyscan gitlab.hpi.de >> ~/.ssh/known_hosts"
    )

    _upload_file(
        client=client,
        path="requirements.txt",
        temp_path="requirements-tmp.txt",
        dest_path="hypex/requirements.txt",
        callback=_run_command_callback(
            client=client,
            install_cmd="~/hypex/.venv/bin/python -m pip install -r ~/hypex/requirements.txt",
        ),
    )

    _logger.info("Now installing R on remote hosts")
    _run_command(
        client=client,
        command="apt-get install -y r-base r-base-core r-recommended",
    )

    _upload_file(
        client=client,
        path="requirements.R",
        temp_path="requirements-tmp.R",
        dest_path="hypex/requirements.R",
        callback=_install_callback(
            install_cmd="RScript requirements.R",
        ),
    )


def install_dependencies(hosts: t.List[str]):
    client = ParallelSSHClient(hosts)
    _install_dependencies(client=client)
