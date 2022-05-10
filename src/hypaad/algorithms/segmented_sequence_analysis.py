from .algorithm_executor import AlgorithmExecutor

__all__ = ["ssa"]


def ssa() -> AlgorithmExecutor:
    return AlgorithmExecutor(
        image_name="ghcr.io/mschroederi/ssa:latest",
        postprocess=None,
    )
