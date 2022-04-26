from .algorithm_executor import AlgorithmExecutor

__all__ = ["ssa"]


def ssa() -> AlgorithmExecutor:
    return AlgorithmExecutor(
        image_name="sopedu:5000/akita/ssa",
        postprocess=None,
    )
