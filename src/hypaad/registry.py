import logging
import typing as t

# pylint: disable=cyclic-import
import hypaad

__all__ = ["Registry"]


class Registry:
    def __init__(
        self, algorithms: t.Dict[str, "hypaad.AlgorithmExecutor"]
    ) -> None:
        self.algorithms = algorithms
        self._logger = logging.getLogger(self.__class__.__name__)

    def register_algorithm(self, algorithm: "hypaad.AlgorithmExecutor") -> None:
        if self.has_algorithm(algorithm.name):
            self._logger.warning("Overriding algorithm %s", algorithm.name)
        self.algorithms[algorithm.name] = algorithm

    def has_algorithm(self, name: str) -> bool:
        return name in self.algorithms

    def get_algorithm(self, name: str) -> "hypaad.AlgorithmExecutor":
        if not self.has_algorithm(name):
            raise ValueError(
                f"No such algorithm {name}. Make sure to correctly register the algorithm."
            )

        return self.algorithms.get(name)

    @classmethod
    def default(cls):
        return Registry(algorithms={"series2graph": hypaad.series2graph()})
