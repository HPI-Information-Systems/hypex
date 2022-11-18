import json
import logging
import subprocess
import typing as t
from pathlib import Path
from time import sleep

import numpy as np
from durations import Duration
from requests.exceptions import ReadTimeout

import docker
import hypex

__all__ = ["AlgorithmExecutor", "AlgorithmRuntimeException"]

DATASET_MOUNT_PATH = "/data"
RESULTS_MOUNT_PATH = "/results"
SCORES_FILE_NAME = "docker-algorithm-scores.csv"
MODEL_FILE_NAME = "model.pkl"

LAUNCH_RETRY_LIMIT = 3
DOCKER_RETRY_LIMIT = 3


class AlgorithmRuntimeException(Exception):
    pass


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


class AlgorithmExecutor:
    client = docker.from_env(timeout=None)

    def __init__(
        self,
        image_name: str,
        default_params: t.Dict[str, t.Any],
        get_timeeval_params: t.Callable[[t.Any], t.List[t.Dict[str, t.Any]]],
        parameter_space: "hypex.MultidimensionalParameterDistribution" = None,
        postprocess: t.Optional[
            t.Callable[[np.array, t.Dict[str, t.Any]], np.array]
        ] = None,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self.image_name = image_name
        self.default_params = default_params
        self.get_timeeval_params = get_timeeval_params
        self.parameter_space = parameter_space
        self.postprocess = postprocess

    @classmethod
    def _get_results_path(cls, args: t.Dict[str, t.Any]) -> Path:
        return Path(args.get("results_path", "./results"))

    @staticmethod
    def _get_uid() -> str:
        uid = subprocess.run(
            ["id", "-u"], capture_output=True, text=True, check=False
        ).stdout.strip()
        if uid == "0":  # if uid is root (0), we don't want to change it
            return ""
        return uid

    def _start_container(self, dataset_path: str, args: t.Dict[str, t.Any]):
        _dataset_path = Path(dataset_path)

        algorithm_args = {
            "dataInput": str(
                (Path(DATASET_MOUNT_PATH) / _dataset_path.name).absolute()
            ),
            "dataOutput": str(
                (Path(RESULTS_MOUNT_PATH) / SCORES_FILE_NAME).absolute()
            ),
            "modelInput": str(
                (Path(RESULTS_MOUNT_PATH) / MODEL_FILE_NAME).absolute()
            ),
            "modelOutput": str(
                (Path(RESULTS_MOUNT_PATH) / MODEL_FILE_NAME).absolute()
            ),
            "executionType": args.get("executionType", "execute"),
            "customParameters": args.get("hyper_params", {}),
        }

        uid = AlgorithmExecutor._get_uid()

        launch_retry_count = 0
        error: docker.errors.ContainerError = None
        while launch_retry_count < LAUNCH_RETRY_LIMIT:
            launch_retry_count += 1
            try:
                self.client.containers.run(
                    image=self.image_name,
                    command=f"execute-algorithm '{json.dumps(algorithm_args, cls=NumpyEncoder)}'",
                    volumes={
                        str(Path(dataset_path).parent.absolute()): {
                            "bind": DATASET_MOUNT_PATH,
                            "mode": "ro",
                        },
                        str(self._get_results_path(args=args).absolute()): {
                            "bind": RESULTS_MOUNT_PATH,
                            "mode": "rw",
                        },
                    },
                    environment={"LOCAL_UID": uid},
                    nano_cpus=int(0.9 * 1e9),  # 0.9 CPU
                    stderr=True,
                    detach=False,
                    remove=True,
                )
                return
            except docker.errors.ContainerError as e:
                self._logger.error(
                    "Docker container launch failed with error %s", e
                )
                error = e
                sleep(30)

        self._logger.info("###############################\n")
        self._logger.error(
            "Docker algorithm failed with status code %d", error.exit_status
        )
        self._logger.info("#### Docker container logs ####")
        self._logger.info(error.stderr)

        raise AlgorithmRuntimeException(error)

    # def _wait(
    #     self, container: "docker.models.Container", args: t.Dict[str, t.Any]
    # ):
    #     @hypex.timeout(seconds=60)
    #     def _get_logs():
    #         return container.logs().decode("utf-8")

    #     result = None

    #     # timeout = Duration(args.get("timeout", "10 minutes"))
    #     retry_count = 0
    #     while retry_count < DOCKER_RETRY_LIMIT:
    #         retry_count += 1
    #         try:
    #             result = container.wait()
    #         # pylint: disable=broad-except
    #         except Exception as e:
    #             self._logger.error(
    #                 "Execution of Docker container failed with error %s [retry_count=%d]", e, retry_count
    #             )
    #             sleep(10)

    #     logs = _get_logs()
    #     self._logger.info("###############################\n")
    #     self._logger.info("Now stopping the container...")
    #     container.stop()
    #     self._logger.info("Now removing the container...")
    #     container.remove()
    #     self._logger.info("Container cleanup completed")

    #     status_code = result["StatusCode"] if result is not None else 1
    #     if status_code != 0:
    #         self._logger.error(
    #             "Docker algorithm failed with status code %d", status_code
    #         )
    #         self._logger.info("#### Docker container logs ####")
    #         self._logger.info(logs)
    #         raise AlgorithmRuntimeException(
    #             f"Docker container returned non-zero status code {status_code}."
    #         )
    #     self._logger.info(
    #         "Docker algorithm completed with status code %d", status_code
    #     )

    def _read_results(self, args: t.Dict[str, t.Any]) -> np.ndarray:
        return np.genfromtxt(
            self._get_results_path(args=args) / SCORES_FILE_NAME, delimiter=","
        )

    def _postprocess(
        self, scores: np.array, args: t.Dict[str, t.Any]
    ) -> np.array:
        if self.postprocess is not None:
            self._logger.info("Now postprocessing the algorithm's results...")
            return self.postprocess(scores, args)

        self._logger.info("No postprocessing steps defined")
        return scores

    def execute(self, dataset_path: str, args: t.Dict[str, t.Any]) -> np.array:
        self._start_container(dataset_path=dataset_path, args=args)
        # self._wait(container=container, args=args)
        results = self._read_results(args=args)

        return self._postprocess(results, args)
