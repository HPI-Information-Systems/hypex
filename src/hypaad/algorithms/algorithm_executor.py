import json
import logging
import subprocess
import typing as t
from pathlib import Path

import docker
import numpy as np
from durations import Duration

__all__ = ["AlgorithmExecutor"]

DATASET_MOUNT_PATH = "/data"
RESULTS_MOUNT_PATH = "/results"
SCORES_FILE_NAME = "docker-algorithm-scores.csv"
MODEL_FILE_NAME = "model.pkl"


class AlgorithmExecutor:
    def __init__(
        self,
        image_name: str,
        postprocess: t.Optional[
            t.Callable[[np.array, t.Dict[str, t.Any]], np.array]
        ] = None,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self.image_name = image_name
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
        client = docker.from_env()
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

        return client.containers.run(
            image=self.image_name,
            command=f"execute-algorithm '{json.dumps(algorithm_args)}'",
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
            detach=True,
        )

    def _wait(
        self, container: "docker.models.Container", args: t.Dict[str, t.Any]
    ):
        timeout = Duration(args.get("timeout", "10 minutes"))
        try:
            result = container.wait(timeout=timeout.to_seconds())
        # pylint: disable=broad-except
        except Exception as e:
            self._logger.error(
                "Execution of Docker container failed with error %s", e
            )
        finally:
            self._logger.info("\n#### Docker container logs ####")
            self._logger.info(container.logs().decode("utf-8"))
            self._logger.info("###############################\n")
            container.stop()

        status_code = result["StatusCode"]

        if status_code != 0:
            self._logger.error(
                "Docker algorithm failed with status code %d", status_code
            )
            raise ValueError(
                f"Docker container returned non-zero status code {status_code}."
            )

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
        container = self._start_container(dataset_path=dataset_path, args=args)
        self._wait(container=container, args=args)
        results = self._read_results(args=args)

        return self._postprocess(results, args)
