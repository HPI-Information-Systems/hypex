import _thread as thread
import logging
import sys
import threading
from pathlib import Path
from time import sleep

import docker
import requests

__all__ = ["timeout", "docker_prune", "docker_prune_cleanup"]

_logger = logging.getLogger(__name__)


def quit_function(fn_name: str):
    _logger.error("%s took too long", fn_name)
    sys.stderr.flush()
    thread.interrupt_main()  # raises KeyboardInterrupt


def timeout(seconds: int):
    """
    Use as decorator to exit process if
    function takes longer than ``seconds`` seconds
    """

    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(seconds, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result

        return inner

    return outer


DOCKER_PRUNE_LOCK_PATH = Path("docker") / "prune.lock"


def docker_prune():
    DOCKER_PRUNE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(DOCKER_PRUNE_LOCK_PATH, "x"):
            client = docker.from_env()
            client.containers.prune()
    except FileExistsError:
        _logger.info(
            "Could not aquire lock. There must be another worker pruning docker containers. "
            "Thus skipping the pruning."
        )
    except requests.exceptions.ReadTimeout:
        _logger.warning(
            "Could not prune containers due to Docker API timeout."
            "Thus skipping the pruning."
        )
        sleep(5)


def docker_prune_cleanup():
    if DOCKER_PRUNE_LOCK_PATH.exists():
        DOCKER_PRUNE_LOCK_PATH.unlink(missing_ok=True)
